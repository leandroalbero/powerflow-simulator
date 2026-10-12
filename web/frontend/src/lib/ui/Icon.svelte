<script lang="ts" context="module">
  // 24×24 stroke icons, drawn for this app.
  const PATHS = {
    sun: 'M12 4V2M12 22v-2M4 12H2M22 12h-2M5.6 5.6 4.2 4.2M19.8 19.8l-1.4-1.4M5.6 18.4l-1.4 1.4M19.8 4.2l-1.4 1.4M12 7a5 5 0 1 0 0 10 5 5 0 0 0 0-10Z',
    moon: 'M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5Z',
    monitor: 'M3 5h18v11H3zM8 20h8M12 16v4',
    sliders: 'M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0M14 4v4M8 10v4M16 16v4',
    play: 'M7 5v14l11-7L7 5Z',
    x: 'M6 6l12 12M18 6 6 18',
    chevronDown: 'm6 9 6 6 6-6',
    check: 'm5 12.5 4.5 4.5L19 7.5',
    refresh: 'M20 11a8 8 0 0 0-14.5-4.5L4 8M4 4v4h4M4 13a8 8 0 0 0 14.5 4.5L20 16M20 20v-4h-4',
    download: 'M12 4v11M7 10.5l5 5 5-5M5 20h14',
    info: 'M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20ZM12 11v6M12 7.5v.01',
    alert: 'M12 3 2 20h20L12 3ZM12 10v4.5M12 17.5v.01',
    plus: 'M12 5v14M5 12h14',
    trash: 'M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3',
    more: 'M5 12h.01M12 12h.01M19 12h.01',
    reset: 'M4 4v6h6M4.5 10A8 8 0 1 1 6 17',
    stop: 'M7 7h10v10H7z',
    battery: 'M3 8h15v8H3zM21 11v2M6 11v2',
    clock: 'M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20ZM12 7v5l3 2',
    database: 'M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3ZM4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3',
    arrowUp: 'M12 19V5M6 11l6-6 6 6',
    arrowDown: 'M12 5v14M6 13l6 6 6-6',
    sort: 'M8 9l4-4 4 4M8 15l4 4 4-4',
  } as const;
  export type IconName = keyof typeof PATHS;
</script>

<script lang="ts">
  export let name: IconName;
  export let size = 16;
  export let stroke = 1.75;
</script>

<svg
  class="icon"
  width={size}
  height={size}
  viewBox="0 0 24 24"
  fill="none"
  stroke="currentColor"
  stroke-width={stroke}
  stroke-linecap="round"
  stroke-linejoin="round"
  aria-hidden="true"
  focusable="false"
>
  <path d={PATHS[name]} fill={name === 'play' || name === 'stop' ? 'currentColor' : 'none'} />
</svg>

<style>
  .icon {
    flex: none;
    display: block;
  }
</style>
