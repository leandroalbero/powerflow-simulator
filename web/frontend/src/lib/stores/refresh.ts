import { get, writable } from 'svelte/store';
import { errorMessage, getRefreshJob, isUnavailable, startDataRefresh } from '../api/client';
import type { RefreshJob } from '../types';
import { refreshDataInfo, setCapability } from './app';
import { toast } from './toasts';

export const refreshJob = writable<RefreshJob | null>(null);

let timer: ReturnType<typeof setTimeout> | null = null;

async function pollJob(jobId: string): Promise<void> {
  try {
    const job = await getRefreshJob(jobId);
    refreshJob.set(job);
    if (job.status === 'running') {
      timer = setTimeout(() => void pollJob(jobId), 1500);
      return;
    }
    if (job.status === 'completed') {
      toast('success', 'Data refreshed', job.message || undefined);
      await refreshDataInfo();
    } else toast('error', 'Data refresh failed', job.error ?? job.message);
  } catch (err) {
    refreshJob.set(null);
    toast('error', 'Lost track of the data refresh', errorMessage(err));
  }
}

export async function startRefresh(full = false): Promise<void> {
  const cur = get(refreshJob);
  if (cur?.status === 'running') return;
  if (timer) clearTimeout(timer);
  try {
    const { job_id } = await startDataRefresh(full);
    refreshJob.set({ job_id, status: 'running', progress_pct: 0, message: 'Starting' });
    void pollJob(job_id);
  } catch (err) {
    if (isUnavailable(err)) {
      setCapability('refresh', 'unavailable');
      toast('info', 'Data refresh needs a backend update');
    } else toast('error', 'Could not start the data refresh', errorMessage(err));
  }
}
