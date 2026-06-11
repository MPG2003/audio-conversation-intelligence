'use client';

import { useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { Bell, CalendarDays, CheckCircle2, Clock3, Filter, Loader2, Rows3, Search, UserRound } from 'lucide-react';
import { apiService, FollowUpAlert } from '@/services/api';
import { useAppStore } from '@/store/useAppStore';
import { cn } from '@/lib/utils';

const priorities = [
  { value: 'All', label: 'Any priority' },
  { value: 'High', label: 'High' },
  { value: 'Medium', label: 'Medium' },
  { value: 'Low', label: 'Low' },
];
const statuses = [
  { value: 'All', label: 'Any status' },
  { value: 'Pending', label: 'Pending' },
  { value: 'Completed', label: 'Completed' },
];
const dateFilters = [
  { value: 'All', label: 'Any date' },
  { value: 'Today', label: 'Today' },
  { value: '7d', label: 'Last 7 days' },
  { value: '30d', label: 'Last 30 days' },
];
const rowOptions = ['5', '10', '25', 'All'];

const priorityClass: Record<string, string> = {
  High: 'bg-red-500/10 text-red-600 border-red-500/20 dark:text-red-400',
  Medium: 'bg-amber-500/10 text-amber-600 border-amber-500/20 dark:text-amber-400',
  Low: 'bg-emerald-500/10 text-emerald-600 border-emerald-500/20 dark:text-emerald-400',
};

function formatDate(value: string) {
  if (!value) return '';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date);
}

function StatusBadge({ status }: { status: FollowUpAlert['status'] }) {
  const isPending = status === 'Pending';
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-bold',
        isPending
          ? 'border-blue-500/20 bg-blue-500/10 text-blue-600 dark:text-blue-400'
          : 'border-emerald-500/20 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400'
      )}
    >
      {isPending ? <Clock3 className="h-3.5 w-3.5" /> : <CheckCircle2 className="h-3.5 w-3.5" />}
      {status}
    </span>
  );
}

function PriorityBadge({ priority }: { priority: FollowUpAlert['priority'] }) {
  return (
    <span className={cn('inline-flex rounded-full border px-2.5 py-1 text-xs font-bold', priorityClass[priority])}>
      {priority}
    </span>
  );
}

function isWithinDateFilter(value: string, filter: string) {
  if (filter === 'All') return true;
  const created = new Date(value);
  if (Number.isNaN(created.getTime())) return false;

  const now = new Date();
  if (filter === 'Today') {
    return created.toDateString() === now.toDateString();
  }

  const days = filter === '7d' ? 7 : 30;
  const cutoff = new Date(now);
  cutoff.setDate(now.getDate() - days);
  return created >= cutoff;
}

function ExpandableText({ text, quoted = false, maxChars = 110 }: { text: string; quoted?: boolean; maxChars?: number }) {
  const [isOpen, setIsOpen] = useState(false);
  const cleanText = text || '-';
  const needsToggle = cleanText.length > maxChars;
  const visibleText = !needsToggle || isOpen ? cleanText : `${cleanText.slice(0, maxChars).trim()}...`;

  return (
    <div>
      <p className="text-sm leading-6 text-gray-600 dark:text-gray-300">
        {quoted ? `"${visibleText}"` : visibleText}
      </p>
      {needsToggle && (
        <button
          onClick={() => setIsOpen((current) => !current)}
          className="mt-1 text-xs font-bold text-blue-600 hover:text-blue-700 dark:text-blue-400"
        >
          {isOpen ? 'Show less' : 'Show more'}
        </button>
      )}
    </div>
  );
}

export function FollowUpAlertsSection() {
  const followUpRefreshKey = useAppStore((state) => state.followUpRefreshKey);
  const refreshFollowUpAlerts = useAppStore((state) => state.refreshFollowUpAlerts);
  const [alerts, setAlerts] = useState<FollowUpAlert[]>([]);
  const [priority, setPriority] = useState('All');
  const [status, setStatus] = useState('Pending');
  const [dateFilter, setDateFilter] = useState('All');
  const [rowsPerPage, setRowsPerPage] = useState('10');
  const [customerName, setCustomerName] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [updatingId, setUpdatingId] = useState<string | null>(null);

  const filteredAlerts = useMemo(
    () => alerts.filter((alert) => isWithinDateFilter(alert.created_date, dateFilter)),
    [alerts, dateFilter]
  );
  const visibleAlerts = useMemo(
    () => rowsPerPage === 'All' ? filteredAlerts : filteredAlerts.slice(0, Number(rowsPerPage)),
    [filteredAlerts, rowsPerPage]
  );
  const activeCount = useMemo(() => filteredAlerts.filter((alert) => alert.status === 'Pending').length, [filteredAlerts]);

  useEffect(() => {
    let isMounted = true;
    setIsLoading(true);
    apiService
      .getFollowUpAlerts({
        priority: priority === 'All' ? undefined : priority,
        status: status === 'All' ? undefined : status,
        customerName: customerName.trim() || undefined,
      })
      .then((items) => {
        if (isMounted) setAlerts(items);
      })
      .catch((error) => {
        console.error('Failed to load follow-up alerts', error);
      })
      .finally(() => {
        if (isMounted) setIsLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, [priority, status, customerName, followUpRefreshKey]);

  const completeAlert = async (alertId: string) => {
    setUpdatingId(alertId);
    try {
      await apiService.updateFollowUpStatus(alertId, 'Completed');
      refreshFollowUpAlerts();
    } catch (error) {
      console.error('Failed to complete follow-up alert', error);
    } finally {
      setUpdatingId(null);
    }
  };

  return (
    <section id="follow-up-alerts" className="py-24 px-4 sm:px-8 relative max-w-7xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: '-100px' }}
        transition={{ duration: 0.6 }}
        className="mb-10 flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between"
      >
        <div>
          <div className="mb-5 inline-flex h-14 w-14 items-center justify-center rounded-2xl bg-blue-500/10">
            <Bell className="h-7 w-7 text-blue-600 dark:text-blue-400" />
          </div>
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white md:text-5xl">
            {'\u{1F514} Follow-Up Alerts'}
          </h2>
          <p className="mt-3 max-w-2xl text-gray-500 dark:text-gray-400">
            Customer commitments, callbacks, demos, proposals, clarifications, and deferred decisions captured as CRM tasks.
          </p>
        </div>

        <div className="grid grid-cols-2 gap-3 sm:flex">
          <div className="rounded-xl border border-gray-200 bg-white/80 px-4 py-3 dark:border-gray-800 dark:bg-gray-950/80">
            <span className="block text-xs font-bold uppercase tracking-wide text-gray-500">Pending</span>
            <strong className="text-2xl text-gray-900 dark:text-white">{activeCount}</strong>
          </div>
          <div className="rounded-xl border border-gray-200 bg-white/80 px-4 py-3 dark:border-gray-800 dark:bg-gray-950/80">
            <span className="block text-xs font-bold uppercase tracking-wide text-gray-500">Showing</span>
            <strong className="text-2xl text-gray-900 dark:text-white">{filteredAlerts.length}</strong>
          </div>
        </div>
      </motion.div>

      <div className="mb-5 grid gap-3 rounded-2xl border border-gray-200 bg-white/80 p-3 shadow-sm dark:border-gray-800 dark:bg-gray-950/80 xl:grid-cols-[1fr_auto_auto_auto_auto]">
        <label className="relative flex items-center">
          <Search className="pointer-events-none absolute left-3 h-4 w-4 text-gray-400" />
          <input
            value={customerName}
            onChange={(event) => setCustomerName(event.target.value)}
            placeholder="Filter customer"
            className="h-11 w-full rounded-xl border border-gray-200 bg-white pl-10 pr-3 text-sm text-gray-900 outline-none transition focus:border-blue-500 dark:border-gray-800 dark:bg-gray-900 dark:text-white"
          />
        </label>

        <div className="flex items-center gap-2 overflow-x-auto">
          <Filter className="h-4 w-4 shrink-0 text-gray-400" />
          {priorities.map((item) => (
            <button
              key={item.value}
              onClick={() => setPriority(item.value)}
              className={cn(
                'h-10 rounded-xl border px-3 text-sm font-bold transition',
                priority === item.value
                  ? 'border-blue-500 bg-blue-500 text-white'
                  : 'border-gray-200 bg-white text-gray-600 hover:border-blue-300 dark:border-gray-800 dark:bg-gray-900 dark:text-gray-300'
              )}
            >
              {item.label}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-2 overflow-x-auto">
          {statuses.map((item) => (
            <button
              key={item.value}
              onClick={() => setStatus(item.value)}
              className={cn(
                'h-10 rounded-xl border px-3 text-sm font-bold transition',
                status === item.value
                  ? 'border-gray-900 bg-gray-900 text-white dark:border-white dark:bg-white dark:text-gray-900'
                  : 'border-gray-200 bg-white text-gray-600 hover:border-gray-400 dark:border-gray-800 dark:bg-gray-900 dark:text-gray-300'
              )}
            >
              {item.label}
            </button>
          ))}
        </div>

        <label className="relative flex items-center">
          <CalendarDays className="pointer-events-none absolute left-3 h-4 w-4 text-gray-400" />
          <select
            value={dateFilter}
            onChange={(event) => setDateFilter(event.target.value)}
            className="h-10 min-w-[150px] appearance-none rounded-xl border border-gray-200 bg-white pl-10 pr-4 text-sm font-bold text-gray-700 outline-none transition focus:border-blue-500 dark:border-gray-800 dark:bg-gray-900 dark:text-gray-200"
          >
            {dateFilters.map((item) => (
              <option key={item.value} value={item.value}>{item.label}</option>
            ))}
          </select>
        </label>

        <label className="relative flex items-center">
          <Rows3 className="pointer-events-none absolute left-3 h-4 w-4 text-gray-400" />
          <select
            value={rowsPerPage}
            onChange={(event) => setRowsPerPage(event.target.value)}
            className="h-10 min-w-[120px] appearance-none rounded-xl border border-gray-200 bg-white pl-10 pr-4 text-sm font-bold text-gray-700 outline-none transition focus:border-blue-500 dark:border-gray-800 dark:bg-gray-900 dark:text-gray-200"
          >
            {rowOptions.map((item) => (
              <option key={item} value={item}>{item === 'All' ? 'All rows' : `${item} rows`}</option>
            ))}
          </select>
        </label>
      </div>

      {!isLoading && filteredAlerts.length > 0 && (
        <p className="mb-3 text-sm text-gray-500">
          Showing {visibleAlerts.length} of {filteredAlerts.length} alert{filteredAlerts.length === 1 ? '' : 's'}
        </p>
      )}

      <div className="overflow-hidden rounded-2xl border border-gray-200 bg-white/90 shadow-xl dark:border-gray-800 dark:bg-gray-950/90">
        {isLoading ? (
          <div className="flex h-64 items-center justify-center gap-3 text-gray-500">
            <Loader2 className="h-6 w-6 animate-spin text-blue-500" />
            Loading alert queue...
          </div>
        ) : filteredAlerts.length === 0 ? (
          <div className="flex h-64 flex-col items-center justify-center gap-3 px-6 text-center text-gray-500">
            <Bell className="h-10 w-10 text-gray-300" />
            <p>No follow-up alerts match the current filters.</p>
          </div>
        ) : (
          <>
            <div className="hidden lg:block">
              <table className="w-full table-fixed">
                <thead className="bg-gray-50 text-left text-xs font-bold uppercase tracking-wide text-gray-500 dark:bg-gray-900/80">
                  <tr>
                    <th className="w-[12%] px-4 py-4">Customer Name</th>
                    <th className="w-[12%] px-4 py-4">Company Name</th>
                    <th className="w-[18%] px-4 py-4">Action Needed</th>
                    <th className="w-[8%] px-4 py-4">Priority</th>
                    <th className="w-[17%] px-4 py-4">Reason</th>
                    <th className="w-[18%] px-4 py-4">Source Statement</th>
                    <th className="w-[10%] px-4 py-4">Created Date</th>
                    <th className="w-[10%] px-4 py-4">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100 dark:divide-gray-800">
                  {visibleAlerts.map((alert) => (
                    <tr key={alert.id} className="align-top transition hover:bg-blue-50/40 dark:hover:bg-blue-950/20">
                      <td className="px-4 py-4">
                        <div className="flex items-center gap-2 font-semibold text-gray-900 dark:text-white">
                          <UserRound className="h-4 w-4 text-gray-400" />
                          {alert.customer_name || 'Unknown'}
                        </div>
                      </td>
                      <td className="px-4 py-4 text-sm text-gray-600 dark:text-gray-300">{alert.company_name || '-'}</td>
                      <td className="px-4 py-4 text-sm font-semibold text-gray-900 dark:text-white">
                        <ExpandableText text={alert.action_needed} maxChars={85} />
                      </td>
                      <td className="px-4 py-4"><PriorityBadge priority={alert.priority} /></td>
                      <td className="px-4 py-4"><ExpandableText text={alert.reason} maxChars={95} /></td>
                      <td className="px-4 py-4"><ExpandableText text={alert.source_text} quoted maxChars={100} /></td>
                      <td className="px-4 py-4 text-sm text-gray-500">{formatDate(alert.created_date)}</td>
                      <td className="px-4 py-4">
                        <div className="flex flex-col items-start gap-2">
                          <StatusBadge status={alert.status} />
                          {alert.status === 'Pending' && (
                            <button
                              onClick={() => completeAlert(alert.id)}
                              disabled={updatingId === alert.id}
                              className="inline-flex h-8 items-center gap-1.5 rounded-lg bg-emerald-600 px-2.5 text-xs font-bold text-white transition hover:bg-emerald-700 disabled:opacity-60"
                            >
                              {updatingId === alert.id ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <CheckCircle2 className="h-3.5 w-3.5" />}
                              Complete
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="grid gap-3 p-3 lg:hidden">
              {visibleAlerts.map((alert) => (
                <article key={alert.id} className="rounded-xl border border-gray-200 bg-white p-4 dark:border-gray-800 dark:bg-gray-900">
                  <div className="mb-3 flex items-start justify-between gap-3">
                    <div>
                      <p className="text-sm font-bold text-gray-900 dark:text-white">{alert.customer_name || 'Unknown Customer'}</p>
                      <p className="text-xs text-gray-500">{alert.company_name || 'No company captured'}</p>
                    </div>
                    <PriorityBadge priority={alert.priority} />
                  </div>
                  <div className="mb-2 font-semibold text-gray-900 dark:text-white">
                    <ExpandableText text={alert.action_needed} maxChars={90} />
                  </div>
                  <div className="mb-3">
                    <ExpandableText text={alert.reason} maxChars={120} />
                  </div>
                  <div className="mb-4 rounded-lg bg-gray-50 p-3 dark:bg-gray-950">
                    <ExpandableText text={alert.source_text} quoted maxChars={130} />
                  </div>
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <span className="text-xs text-gray-500">{formatDate(alert.created_date)}</span>
                    <div className="flex items-center gap-2">
                      <StatusBadge status={alert.status} />
                      {alert.status === 'Pending' && (
                        <button
                          onClick={() => completeAlert(alert.id)}
                          disabled={updatingId === alert.id}
                          className="inline-flex h-9 items-center gap-1.5 rounded-lg bg-emerald-600 px-3 text-xs font-bold text-white transition hover:bg-emerald-700 disabled:opacity-60"
                        >
                          {updatingId === alert.id ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <CheckCircle2 className="h-3.5 w-3.5" />}
                          Complete
                        </button>
                      )}
                    </div>
                  </div>
                </article>
              ))}
            </div>
          </>
        )}
      </div>
    </section>
  );
}
