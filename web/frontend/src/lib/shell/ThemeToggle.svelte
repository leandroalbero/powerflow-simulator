<script lang="ts">
  import { cycleTheme, themePref } from '../stores/theme';
  import Icon from '../ui/Icon.svelte';

  const LABEL = { system: 'Theme: match system', light: 'Theme: light', dark: 'Theme: dark' } as const;
  const ICON = { system: 'monitor', light: 'sun', dark: 'moon' } as const;
</script>

<button type="button" class="toggle" on:click={cycleTheme} aria-label="{LABEL[$themePref]}. Click to change." title={LABEL[$themePref]}>
  <Icon name={ICON[$themePref]} size={17} />
</button>

<style>
  .toggle {
    display: grid;
    place-items: center;
    width: 36px;
    height: 36px;
    border-radius: var(--r-md);
    border: 0;
    background: transparent;
    color: var(--text-2);
  }
  .toggle:hover {
    background: var(--surface-3);
    color: var(--text);
  }
  @media (max-width: 720px), (pointer: coarse) {
    .toggle {
      width: 40px;
      height: 40px;
    }
  }
</style>
