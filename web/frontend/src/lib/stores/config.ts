import { writable } from 'svelte/store';
import type { DataInfo, SystemConfig } from '../types/index';

/** System configuration (battery, grid, tariff). */
export const systemConfig = writable<SystemConfig | null>(null);

/** Information about loaded solar/load data files. */
export const dataInfo = writable<DataInfo | null>(null);
