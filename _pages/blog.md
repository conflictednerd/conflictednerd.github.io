---
layout: book
prose: true
permalink: /blog/
title: Notes
---

<main id="top">
  <header class="title doc-head">
    <h1>{{ page.title }}</h1>
  </header>

  <section class="sec" id="notes" data-title="{{ page.title }}">
    <div class="body">
      <aside class="mn note" id="toc-note">{{ site.blog_description }}</aside>
      {% include book/notes.liquid %}
    </div>
  </section>
</main>
