---
layout: book
prose: true
permalink: /publications/
title: Publications
description: publications in reversed chronological order
---

<main id="top">
  <header class="title doc-head">
    <h1>{{ page.title }}</h1>
  </header>

  <section class="sec" id="publications" data-title="{{ page.title }}">
    <div class="body pubs">{% bibliography --template book-bib --group_by none %}</div>
  </section>
</main>
