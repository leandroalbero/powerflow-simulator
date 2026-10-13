<script lang="ts">
  export let checked: boolean;
  export let label: string;
  export let description: string | null = null;
  export let disabled = false;
  export let id = `sw-${Math.random().toString(36).slice(2, 8)}`;
</script>

<div class="row" class:disabled>
  <div class="text">
    <label for={id}>{label}</label>
    {#if description}<p class="desc" id="{id}-d">{description}</p>{/if}
  </div>
  <input
    {id}
    type="checkbox"
    role="switch"
    bind:checked
    {disabled}
    aria-describedby={description ? `${id}-d` : undefined}
    on:change
  />
</div>

<style>
  .row {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--s-4);
  }
  /* Dim the control, not the description: it explains why the switch is off and must stay readable. */
  .disabled label,
  .disabled input {
    opacity: 0.55;
  }
  label {
    font-size: var(--fs-md);
    font-weight: 550;
    cursor: pointer;
  }
  .desc {
    margin-top: 2px;
    font-size: var(--fs-xs);
    color: var(--text-3);
  }
  input {
    appearance: none;
    flex: none;
    position: relative;
    width: 38px;
    height: 22px;
    margin: 1px 0 0;
    border-radius: var(--r-pill);
    background: var(--border-strong);
    cursor: pointer;
    transition: background var(--dur) var(--ease);
  }
  input::after {
    content: '';
    position: absolute;
    top: 3px;
    left: 3px;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: #fff;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.25);
    transition: transform var(--dur) var(--ease);
  }
  input:checked {
    background: var(--accent);
  }
  input:checked::after {
    transform: translateX(16px);
  }
  input:disabled {
    cursor: not-allowed;
  }
  @media (max-width: 720px), (pointer: coarse) {
    .row {
      min-height: 40px;
    }
    label {
      display: inline-flex;
      align-items: center;
      min-height: 40px;
      margin: -8px 0;
    }
    /* 40px tall hit area; the visible track stays 26px via the content-box background. */
    input {
      width: 44px;
      height: 40px;
      padding: 7px 0;
      margin: -7px 0;
      background-clip: content-box;
    }
    input::after {
      top: 10px;
      width: 20px;
      height: 20px;
    }
    input:checked::after {
      transform: translateX(18px);
    }
  }
</style>
