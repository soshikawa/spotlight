"""
Spotlight wsgi application
"""

import asyncio
import mimetypes
import multiprocessing.connection
import os
import re
import time
import uuid
from concurrent.futures import CancelledError, Future
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Literal, Optional, Union, cast
import numpy as np

import pandas as pd
from fastapi import Cookie, FastAPI, Request, status
from fastapi.datastructures import Headers
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from httpx import URL, AsyncClient
from typing_extensions import Annotated

from renumics.spotlight import dtypes as spotlight_dtypes
from renumics.spotlight import layouts
from renumics.spotlight.analysis import find_issues
from renumics.spotlight.analysis.typing import DataIssue
from renumics.spotlight.app_config import AppConfig
from renumics.spotlight.backend.apis import plugins as plugin_api
from renumics.spotlight.backend.apis import websocket
from renumics.spotlight.backend.config import Config
from renumics.spotlight.backend.exceptions import Problem
from renumics.spotlight.backend.middlewares.timing import add_timing_middleware
from renumics.spotlight.backend.tasks.task_manager import TaskManager
from renumics.spotlight.backend.websockets import (
    Message,
    RefreshMessage,
    ResetLayoutMessage,
    WebsocketManager,
)
from renumics.spotlight.data_source import DataSource, create_datasource
from renumics.spotlight.data_store import DataStore
from renumics.spotlight.develop.project import get_project_info
from renumics.spotlight.embeddings import create_embedders, run_embedders
from renumics.spotlight.layout.nodes import Layout
from renumics.spotlight.logging import logger
from renumics.spotlight.plugin_loader import load_plugins
from renumics.spotlight.reporting import (
    emit_exception_event,
    emit_exit_event,
    emit_startup_event,
)
from renumics.spotlight.settings import settings
from renumics.spotlight.typing import PathType
from renumics.spotlight.backend.tasks.clustering import compute_hdbscan, compute_leiden, compute_evoc

CURRENT_LAYOUT_KEY = "layout.current"


# explicitly set mimetypes for broken python setups
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("image/svg+xml", ".svg")


class UncachedStaticFiles(StaticFiles):
    """
    FastAPI StaticFiles but without caching
    """

    def is_not_modified(
        self, response_headers: Headers, request_headers: Headers
    ) -> bool:
        """
        Never Cache
        """
        return False


class IssuesUpdatedMessage(Message):
    """
    Notify about updated issues.
    """

    type: Literal["issuesUpdated"] = "issuesUpdated"
    data: Any = None


class ColumnsUpdatedMessage(Message):
    """
    Notify about updated embeddings.
    """

    type: Literal["columnsUpdated"] = "columnsUpdated"
    data: List[str]


class SpotlightApp(FastAPI):
    """
    Spotlight wsgi application
    """

    # lifecycle
    _loop: asyncio.AbstractEventLoop

    # connection
    _connection: multiprocessing.connection.Connection
    _receiver_thread: Thread

    # datasource
    _dataset: Optional[Union[PathType, pd.DataFrame]]
    _user_dtypes: spotlight_dtypes.DTypeMap
    _data_source: Optional[DataSource]
    _data_store: Optional[DataStore]

    task_manager: TaskManager
    websocket_manager: Optional[WebsocketManager]
    _layout: Layout
    config: Config
    username: str
    filebrowsing_allowed: bool

    # dev
    project_root: PathType
    vite_url: Optional[str]

    # data issues
    issues: Optional[List[DataIssue]] = []
    _custom_issues: List[DataIssue] = []
    analyze_columns: Union[List[str], bool]

    # embedding
    embed_columns: Union[List[str], bool]

    # clustering
    cluster_column: Optional[str] = None

    def __init__(self) -> None:
        super().__init__()
        self.task_manager = TaskManager()
        self.websocket_manager = None
        self.config = Config()
        self._layout = layouts.default()
        self.project_root = Path.cwd()
        self.vite_url = None
        self.username = ""
        self.filebrowsing_allowed = False
        self.analyze_columns = False
        self.issues = None
        self._custom_issues = []
        self.embed_columns = False
        self.cluster = None
        self.cluster_column = None
        self.cluster_k = None
        self.cluster_resolution = None
        self.cluster_name = None
        self._dataset = None
        self._user_dtypes = {}
        self._data_source = None
        self._data_store = None

        @self.on_event("startup")
        def _() -> None:
            port = int(os.environ["CONNECTION_PORT"])
            authkey = os.environ["CONNECTION_AUTHKEY"]

            self._loop = asyncio.get_running_loop()

            for _ in range(10):
                try:
                    self._connection = multiprocessing.connection.Client(
                        ("127.0.0.1", port), authkey=authkey.encode()
                    )
                except ConnectionRefusedError:
                    time.sleep(0.1)
                else:
                    break
            else:
                raise RuntimeError("Failed to connect to parent process")

            self._receiver_thread = Thread(target=self._receive, daemon=True)
            self._receiver_thread.start()
            self._connection.send({"kind": "startup"})

            def handle_ws_connect(active_connections: int) -> None:
                self._connection.send(
                    {"kind": "frontend_connected", "data": active_connections}
                )

            def handle_ws_disconnect(active_connections: int) -> None:
                self._connection.send(
                    {"kind": "frontend_disconnected", "data": active_connections}
                )

            self.websocket_manager = WebsocketManager(asyncio.get_running_loop())
            self.websocket_manager.add_connect_callback(handle_ws_connect)
            self.websocket_manager.add_disconnect_callback(handle_ws_disconnect)

            self.vite_url = os.environ.get("VITE_URL")

            emit_startup_event()

        @self.on_event("shutdown")
        def _() -> None:
            self._receiver_thread.join(0.1)
            self.task_manager.shutdown()
            emit_exit_event()

        self.include_router(websocket.router, prefix="/api")
        self.include_router(plugin_api.router, prefix="/api/plugins")


        @self.exception_handler(Exception)
        async def _(request: Request, e: Exception) -> JSONResponse:
            if settings.verbose:
                logger.exception(e)
            else:
                logger.info(e)
            emit_exception_event(request.url.path, self._data_source)
            class_name = type(e).__name__
            title = re.sub(r"([a-z])([A-Z])", r"\1 \2", class_name)
            return JSONResponse(
                {"title": title, "detail": str(e), "type": class_name},
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        @self.exception_handler(Problem)
        async def _(request: Request, problem: Problem) -> JSONResponse:
            if settings.verbose:
                logger.exception(problem)
            else:
                logger.info(problem)
            emit_exception_event(request.url.path, self._data_source)
            return JSONResponse(
                {
                    "title": problem.title,
                    "detail": problem.detail,
                    "type": type(problem).__name__,
                },
                status_code=problem.status_code,
            )

        for plugin in load_plugins():
            plugin.activate(self)

        try:
            # Mount frontend files as uncached,
            # so that we always get the new frontend after updating spotlight.
            # NOTE: we might not need this if we added a version hash
            # to our built js files
            self.mount(
                "/static",
                UncachedStaticFiles(packages=["renumics.spotlight.backend"]),
                name="assets",
            )
        except AssertionError:
            logger.warning("Frontend module is missing. No frontend will be served.")

        templates = Jinja2Templates(
            directory=Path(__file__).parent / "backend" / "templates"
        )

        async def _reverse_proxy(request: Request) -> Response:
            http_server = AsyncClient(base_url=request.app.vite_url)
            url = URL(path=request.url.path, query=request.url.query.encode("utf-8"))

            # URL-encoding is not accepted by vite. Use unencoded path instead.

            url._uri_reference = url._uri_reference._replace(path=request.url.path)

            body = await request.body()

            rp_req = http_server.build_request(
                request.method,
                url,
                headers=request.headers.raw,
                content=body,
            )

            rp_resp = await http_server.send(rp_req, stream=False)

            return Response(
                content=rp_resp.content,
                status_code=rp_resp.status_code,
                headers=rp_resp.headers,
            )

        @self.get("/")
        def _(
            request: Request, browser_id: Annotated[Union[str, None], Cookie()] = None
        ) -> Any:
            response = templates.TemplateResponse(
                request=request,
                name="index.html",
                context={
                    "request": request,
                    "dev": settings.dev,
                    "dev_location": get_project_info().type,
                    "vite_url": request.app.vite_url,
                    "filebrowsing_allowed": request.app.filebrowsing_allowed,
                },
            )
            response.set_cookie(
                "browser_id",
                browser_id or str(uuid.uuid4()),
                samesite="none",
                secure=True,
            )
            return response

        if settings.dev:
            logger.info("Running in dev mode")
            self.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            add_timing_middleware(self)

            # Reverse proxy routes for webworker loading in dev mode
            self.add_route("/src/{path:path}", _reverse_proxy, ["POST", "GET"])
            self.add_route(
                "/node_modules/.vite/dist/client/{path:path}",
                _reverse_proxy,
                ["POST", "GET"],
            )
            self.add_route(
                "/node_modules/.vite/deps/{path:path}", _reverse_proxy, ["POST", "GET"]
            )
            self.add_route(
                "/node_modules/.pnpm/{path:path}", _reverse_proxy, ["POST", "GET"]
            )

    def update(self, config: AppConfig) -> None:
        """
        Update application config.
        """
        try:
            if config.project_root is not None:
                self.project_root = config.project_root
            if config.dtypes is not None:
                self._user_dtypes = config.dtypes
            if config.analyze is not None:
                self.analyze_columns = config.analyze
            if config.custom_issues is not None:
                self.custom_issues = config.custom_issues
            if config.embed is not None:
                self.embed_columns = config.embed
            if config.cluster is not None:
                self.cluster = config.cluster
                self.cluster_column = config.cluster_col #added
                self.cluster_k = config.cluster_k
                self.cluster_resolution = config.cluster_resolution
                self.cluster_name = config.cluster_name
                #logger.info(f'cluster_column initialized: {self.cluster_column}')
            if config.dataset is not None:
                self._dataset = config.dataset
                self._data_source = create_datasource(self._dataset)
            if config.layout is not None:
                self._layout = config.layout or layouts.default()
            if config.filebrowsing_allowed is not None:
                self.filebrowsing_allowed = config.filebrowsing_allowed

            if config.dtypes is not None or config.dataset is not None:
                if self._data_source is not None:
                    self._data_store = DataStore(self._data_source, self._user_dtypes)
                    self._broadcast(RefreshMessage())
                    self._update_issues()
                    self._update_embeddings()
                    if self.cluster_column is not None:
                        self._update_clusters()
            if config.layout is not None:
                if self._data_store is not None:
                    dataset_uid = self._data_store.uid
                    future = asyncio.run_coroutine_threadsafe(
                        self.config.remove_all(CURRENT_LAYOUT_KEY, dataset=dataset_uid),
                        self._loop,
                    )
                    future.result()
                self._broadcast(ResetLayoutMessage())

            for plugin in load_plugins():
                plugin.update(self, config)
        except Exception as e:
            self._connection.send({"kind": "update_complete", "error": e})
            logger.exception(e)

        self._connection.send({"kind": "update_complete"})

    def _handle_message(self, message: Any) -> None:
        kind = message.get("kind")
        data = message.get("data")

        if kind is None:
            logger.error(f"Malformed message from client process:\n\t{message}")
        elif kind == "update":
            self.update(data)
        elif kind == "get_df":
            df = self._data_source.df if self._data_source else None
            self._connection.send({"kind": "df", "data": df})
        elif kind == "refresh_frontends":
            self._broadcast(RefreshMessage())
        else:
            logger.warning(f"Unknown message from client process:\n\t{message}")

    def _receive(self) -> None:
        while True:
            try:
                self._handle_message(self._connection.recv())
            except EOFError:
                # the master process closed the connection
                # just stop receiving
                return

    @property
    def data_store(self) -> Optional[DataStore]:
        """
        Current data store.
        """
        return self._data_store

    @property
    def custom_issues(self) -> List[DataIssue]:
        """
        User supplied `DataIssue`s
        """
        return self._custom_issues

    @custom_issues.setter
    def custom_issues(self, issues: List[DataIssue]) -> None:
        self._custom_issues = issues
        self._broadcast(IssuesUpdatedMessage())

    @property
    def layout(self) -> Layout:
        """
        Frontend layout
        """
        return self._layout

    async def get_current_layout_dict(self, user_id: str) -> Optional[Dict]:
        """
        Get the user's current layout (as dict)
        """

        if not self._data_store:
            return None

        dataset_uid = self._data_store.uid
        layout = await self.config.get(
            CURRENT_LAYOUT_KEY, dataset=dataset_uid, user=user_id
        ) or self.layout.model_dump(by_alias=True)
        return cast(Optional[Dict], layout)

    def _update_issues(self) -> None:
        """
        Update issues and notify client about.
        """

        # early out for [] and False
        if not self.analyze_columns:
            self.issues = []
            self._broadcast(IssuesUpdatedMessage())
            return

        self.issues = None
        self._broadcast(IssuesUpdatedMessage())
        if self._data_store is None:
            return

        if self.analyze_columns is True:
            columns = self._data_store.column_names
        else:
            columns = self.analyze_columns

        task = self.task_manager.create_task(
            find_issues, (self._data_store, columns), name="update_issues"
        )

        def _on_issues_ready(future: Future) -> None:
            try:
                self.issues = future.result()
            except CancelledError:
                return
            self._broadcast(IssuesUpdatedMessage())

        task.future.add_done_callback(_on_issues_ready)

    def _update_embeddings(self) -> None:
        """
        Update embeddings, update them in the data store and notify client about.
        """
        if not self.embed_columns:
            return

        if self._data_store is None:
            return

        logger.info("Embedding started.")

        if self.embed_columns is True:
            embed_columns = self._data_store.column_names
        else:
            embed_columns = [
                column
                for column in self.embed_columns
                if column in self._data_store.column_names
            ]

        embedders = create_embedders(self._data_store, embed_columns)

        self._data_store.embeddings = {column: None for column in embedders}

        task = self.task_manager.create_task(
            run_embedders, (embedders,), name="update_embeddings"
        )

        def _on_embeddings_ready(future: Future) -> None:
            if self._data_store is None:
                return
            try:
                self._data_store.embeddings = future.result()
            except CancelledError:
                return
            logger.info("Embedding done.")
            self._broadcast(ColumnsUpdatedMessage(data=list(embedders.keys())))

        task.future.add_done_callback(_on_embeddings_ready)

    def _update_clusters(self) -> None:
        
        if self._data_store is None:
            return
        # find first embedding column

        #TODO: Fix to allow embedding column specification
        embedding_columns = [
            col for col, dtype in self._data_store.dtypes.items()
            if spotlight_dtypes.is_embedding_dtype(dtype)
        ]

        if not embedding_columns:
            return

        column_name = embedding_columns[0]
        indices = list(range(len(self._data_store)))

        if self.cluster == 'leiden':
            logger.info('leiden initiated')
            task = self.task_manager.create_task(
                compute_leiden,
                (self._data_store, [column_name], indices),
                kwargs={
                    'k': self.cluster_k,
                    'resolution': self.cluster_resolution,
                },
                name="cluster",
            )
        elif self.cluster == 'evoc':
            logger.info('evoc initiated')
            task = self.task_manager.create_task(
                compute_evoc,
                (self._data_store, [column_name], indices),
                name="cluster",
            )
        elif self.cluster == 'hdbscan':
            logger.info('hdbscan initiated')
            task = self.task_manager.create_task(
                compute_hdbscan,
                (self._data_store, [column_name], indices),
                name="cluster",
            )
        

        def _on_cluster_done(future: Future) -> None:
            if self._data_store is None:
                return
            try:
                labels, valid_indices = future.result()
            except CancelledError:
                return
            except Exception as e:
                logger.error(f"Clustering failed: {e}")
                return

            '''
            full_labels = np.full(len(self._data_store), -1, dtype=int)
            full_labels[valid_indices] = labels
            unique_labels = sorted(set(full_labels.tolist()))
            
            '''

            #print(f'cluster_column = {self.cluster_column}')
            #print(f'available columns = {self._data_source._df.columns.tolist()}')

            labels = [label + 1 for label in labels]
             #removes the -1 for outliers in hdbscan. for all other clustering methods, since they dont start at -1 it doesnt affect
            full_labels = np.full(len(self._data_store), 0, dtype=int)
            full_labels[valid_indices] = labels
            unique_labels = sorted(set(full_labels.tolist()))

            categories = []

            #from transformers import AutoProcessor, AutoModelForImageTextToText

            #model_id = "LiquidAI/LFM2.5-VL-450M"
            #model = AutoModelForImageTextToText.from_pretrained(
            #    model_id,
            #    device_map="auto",
            #    dtype="bfloat16"
            #)

            #processor = AutoProcessor.from_pretrained(model_id)

            if self.cluster_name:
                logger.info('clustering started')
                import ollama
                model = ollama.Client(
                    host='http://localhost:11434',
                    headers={'Authorization': 'Bearer ollama'}
                )

                for unique_label in unique_labels:
                    label_idx = [] #stores all indices in cluster
                    
                    if unique_label == 0: 
                        categories.append('Outliers')
                        continue

                    for i in range(len(full_labels)):
                        if full_labels[i] == unique_label:
                            label_idx.append(i)

                    if len(label_idx) > 10:
                        label_idx = np.array(label_idx)
                        summary_idx = np.random.choice(label_idx, 10, replace=False)
                    else:
                        summary_idx = label_idx
                    
                    summary_messages = self._data_source.get_column_values(self.cluster_column, summary_idx)
                    mess_type = self._user_dtypes[self.cluster_column]
                    
                    if mess_type == spotlight_dtypes.str_dtype:
                        prompt = [{
                            "role": "user",
                            "content": f"Generate a common topic between the following texts in less than 5 words. Do not output anything else: {'/'.join(summary_messages)}"
                        }]
                    elif mess_type == spotlight_dtypes.image_dtype:
                        prompt = [{
                            "role": 'user',
                            "content": f"Generate a one or two word summarization of the common topic between the following text: {', '.join(summary_messages)}"
                        }] #TODO: FIX FOR IMAGES


                    response = model.chat(model='llama3.2:3b', messages=prompt, think=False)['message']['content']
                    #input = processor.apply_chat_template(
                    #    prompt,
                    #    add_generation_prompt=True,
                    #    return_tensors="pt",
                    #    return_dict=True,
                    #    tokenize=True
                    #).to(model.device)
                    #outputs = model.generate(**input, max_new_tokens=20)
                    #response = processor.batch_decode(outputs)[0]

                    categories.append(response)

                self._data_store.add_computed_column(
                    "cluster", spotlight_dtypes.CategoryDType(categories), full_labels
                )
            else:
                self._data_store.add_computed_column('cluster_id', spotlight_dtypes.int_dtype, full_labels)

            '''
            unique_labels = [label + 1 for label in unique_labels] 
            categories = [str(label) for label in unique_labels]
            label_to_idx = {categories[i]: i  for i in range(len(categories))}
            categorical_labels = np.array([label_to_idx[str(l)] for l in full_labels], dtype=int)
            
            
            
            categories = ['Outliers' if label=='-1' else label for label in categories]
            
            self._data_store.add_computed_column(
                "cluster", spotlight_dtypes.CategoryDType(categories), categorical_labels
            )
            '''
            #self._broadcast(ColumnsUpdatedMessage(data=["cluster"]))
            self._broadcast(RefreshMessage())

        task.future.add_done_callback(_on_cluster_done)

    def _broadcast(self, message: Message) -> None:
        """
        Broadcast a message to all connected clients via websocket
        """
        if self.websocket_manager:
            self.websocket_manager.broadcast(message)
