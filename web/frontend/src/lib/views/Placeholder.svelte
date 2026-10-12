<script lang="ts">
  export let title: string;
  export let text: string;
</script>

<div class="ph">
  <h1>{title}</h1>
  <p>{text}</p>
</div>

<style>
  .ph {
    padding: var(--s-7) 0;
    max-width: 60ch;
  }
  h1 {
    font-size: var(--fs-2xl);
    font-weight: 680;
    letter-spacing: -0.02em;
    margin-bottom: var(--s-2);
  }
  p {
    color: var(--text-2);
    font-size: var(--fs-lg);
  }
</style>
