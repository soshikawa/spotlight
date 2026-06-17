import { DataKind } from './datatypes';
import _ from 'lodash';
import { Predicate } from './types';

type PredicateRegistry = Record<string, { [key: string]: Predicate }>;

const numberFilters = {
    equal: {
        shorthand: '=',
        compare: (value: number, ref: number) =>
            isNaN(ref) ? isNaN(value) : value === ref,
    },
    unequal: {
        shorthand: '!=',
        compare: (value: number, ref: number) =>
            isNaN(ref) ? !isNaN(value) : value !== ref,
    },
    greater: {
        shorthand: '>',
        compare: (value: number, ref: number) => value > ref,
    },
    lesser: {
        shorthand: '<',
        compare: (value: number, ref: number) => value < ref,
    },
    greaterOrEqual: {
        shorthand: '>=',
        compare: (value: number, ref: number) =>
            isNaN(ref) ? isNaN(value) : value >= ref,
    },
    lesserOrEqual: {
        shorthand: '<=',
        compare: (value: number, ref: number) =>
            isNaN(ref) ? isNaN(value) : value <= ref,
    },
};

// Added to compare datetime
const dateTimeFilters = {
    equal: {
        shorthand: '=',
        compare: (value: string, ref: string) =>
            new Date(value).getTime() === new Date(ref).getTime(),
    },
    unequal: {
        shorthand: '!=',
        compare: (value: string, ref: string) =>
            new Date(value).getTime() !== new Date(ref).getTime(),
    },
    greater: {
        shorthand: '>',
        compare: (value: string, ref: string) =>
            new Date(value).getTime() > new Date(ref).getTime(),
    },
    lesser: {
        shorthand: '<',
        compare: (value: string, ref: string) =>
            new Date(value).getTime() < new Date(ref).getTime(),
    },
    greaterOrEqual: {
        shorthand: '>=',
        compare: (value: string, ref: string) =>
            new Date(value).getTime() >= new Date(ref).getTime(),
    },
    lesserOrEqual: {
        shorthand: '<=',
        compare: (value: string, ref: string) =>
            new Date(value).getTime() <= new Date(ref).getTime(),
    },
};

const refFilters = {
    equal: {
        shorthand: '=',
        compare: (value: string | null, ref: string | null) =>
            value === null || ref === null
                ? value === ref
                : matchString(value, ref as string),
    },
    unequal: {
        shorthand: '!=',
        compare: (value: string | null, ref: string | null) =>
            value === null || ref === null
                ? value !== ref
                : !matchString(value, ref as string),
    },
};

function matchString(value: string, ref: string) {
    if (!ref?.length) {
        return !value;
    }
    try {
        return new RegExp(ref).test(value);
    } catch (error) {
        return value.includes(ref);
    }
}

const predicatesByType: PredicateRegistry = {
    float: numberFilters,
    int: numberFilters,
    datetime: dateTimeFilters, //added to compare datetime
    bool: {
        equal: {
            shorthand: '=',
            compare: (value: boolean, ref: boolean) => value === ref,
        },
    },
    str: {
        equal: {
            shorthand: '=',
            compare: (value: string | null, ref: string) =>
                value !== null && matchString(value, ref),
        },
        unequal: {
            shorthand: '!=',
            compare: (value: string | null, ref: string) =>
                value === null || !matchString(value, ref),
        },
    },
    Category: {
        equal: numberFilters.equal,
        unequal: numberFilters.unequal,
    },
    Audio: refFilters,
    Mesh: refFilters,
    Image: refFilters,
    Video: refFilters,
    Sequence1D: refFilters,
};

export const getApplicablePredicates = (kind: DataKind): Record<string, Predicate> =>
    predicatesByType[kind] || {};

export const hasApplicablePredicates = (kind: DataKind): boolean =>
    _.size(getApplicablePredicates(kind)) > 0;
