# conflictednerd.github.io

Personal site of Saeed Hedayatian, built with [Jekyll](https://jekyllrb.com/) and published by GitHub Actions on every push to `main`.

## Editing content

| To change                               | Edit                                                             |
| --------------------------------------- | ---------------------------------------------------------------- |
| Bio, portrait, education, interests     | `_pages/about.md` (text and front matter)                        |
| Publications                            | `_bibliography/papers.bib`                                       |
| Hills of the landscape on the home page | `_data/plate.yml` (one entry per paper; fields explained inside) |
| Teaching and service                    | `_data/service.yml`                                              |
| Notes (blog posts)                      | `_posts/`, one Markdown file per note                            |
| Footer links                            | `_data/socials.yml`                                              |
| Name, e-mail, description, analytics    | `_config.yml`                                                    |

In a publication entry, the fields beyond standard BibTeX are: `venue` (text shown after the authors), `html`, `code` and `website` (links),
`preview` (an image in `assets/img/publication_preview/`), `preview_alt`, `preview_class` (`heat` for a colour map, omitted for a diagram),
and `annotation` (a margin note such as "\* Equal contribution").

In a note's front matter, `description` is the one-line summary shown in lists, `math: true` loads MathJax, and `toc: true` adds a contents
list in the margin. Display equations need a blank line before and after the `$$ … $$` block.

## How it is put together

- `_layouts/book.liquid` is the page shell (head, running head, footer). `home.liquid`, `post.liquid`, `page.liquid` and `archive.liquid` fill
  it in; `book-bib.liquid` renders one publication.
- `_includes/book/` holds the running head, the footer and the list of notes.
- `assets/book/book.css` styles everything; `prose.css` adds the styles for long-form pages.
- `assets/book/plate.js` draws the landscape on the home page; `book.js` has the theme switch, the running head, the notes margin and the
  busy beaver machine.

## Previewing locally

```sh
bundle install
bundle exec jekyll serve   # then open http://localhost:4000
```
