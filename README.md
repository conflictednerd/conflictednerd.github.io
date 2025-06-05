# User Manual

**Quick reference for maintaining your academic website**

## 📚 Adding Publications

### Basic Publication Entry

Edit `_bibliography/papers.bib` and add entries in this format:

```bibtex
@article{author2024paper,
  title={Your Paper Title},
  author={Hedayatian, Saeed and Other, Author},
  journal={Conference/Journal Name},
  year={2024},
  url={https://arxiv.org/abs/2404.xxxxx},
  html={https://arxiv.org/abs/2404.xxxxx},
  arxiv={2404.xxxxx},
  selected={true}
}
```

### All Available Fields

```bibtex
@article{citation_key,
  title={Paper Title},
  author={Hedayatian, Saeed and Coauthor, Name},
  journal={Venue Name},
  year={2024},
  
  % Links (choose what applies)
  url={https://arxiv.org/abs/XXXX.XXXXX},        % Main link
  html={https://arxiv.org/abs/XXXX.XXXXX},       % Makes title clickable
  arxiv={XXXX.XXXXX},                            % ArXiv ID only
  pdf={paper.pdf},                               % PDF filename or URL
  code={https://github.com/user/repo},           % Code repository
  website={https://project-page.com},            % Project website
  slides={slides.pdf},                           % Presentation slides
  poster={poster.pdf},                           % Poster file
  video={https://youtube.com/watch?v=xxx},       % Video presentation
  
  % Display options
  selected={true},                               % Show on homepage
  preview={thumbnail.png},                       % Thumbnail image
  abstract={Brief description of the paper...},  % Abstract text
  
  % Additional metadata
  doi={10.1000/example},                         % DOI
  isbn={978-0000000000},                         % ISBN
  note={Best Paper Award},                       % Special notes
  award={Best Paper},                            % Award name
  award_name={Outstanding Paper Award}           % Longer award name
}
```

### File Organization

**PDFs**: Upload to `assets/pdf/` folder
**Thumbnails**: Upload to `assets/img/publication_preview/` (400x300px recommended)

### Homepage Display

- Set `selected={true}` to show paper on homepage
- Ensure `selected_papers: true` in `_pages/about.md`
- Papers are automatically grouped by year

## ✍️ Adding Blog Posts

### Create New Post

1. Go to `_posts/` folder
2. Create file named: `YYYY-MM-DD-title.md`
3. Add front matter and content

### Blog Post Template

```markdown
---
layout: post
title: Your Post Title
date: 2024-06-04 15:30:00
description: Brief description for SEO and previews
tags: research machine-learning
categories: blog
related_posts: false
---

Your content here...

## LaTeX Math Support

Inline: $E = mc^2$

Display: $$\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}$$

## Code Blocks

```python
def hello_world():
    print("Hello, World!")
```

## Citations

You can cite papers from your bibliography {% cite hedayatian2024paper %}.
```

### Front Matter Options

```yaml
---
layout: post
title: Post Title
date: 2024-06-04 15:30:00-0800
description: SEO description
tags: tag1 tag2 tag3
categories: blog research
related_posts: true              # Show related posts
toc: true                        # Table of contents
featured: true                   # Feature this post
thumbnail: assets/img/thumb.jpg   # Post thumbnail
---
```

### Migrating Old Posts

1. Copy markdown content from old blog
2. Add proper front matter (see template above)
3. Rename file to `YYYY-MM-DD-title.md` format
4. Update any image paths to `assets/img/`

## 🖼️ Managing Images

### Profile Picture
- Upload to: `assets/img/prof_pic.jpg`
- Recommended: 400x400px, square
- Auto-rotates based on EXIF data

### Blog Post Images
- Upload to: `assets/img/`
- Reference as: `![Alt text]({{ '/assets/img/filename.jpg' | relative_url }})`

### Publication Thumbnails
- Upload to: `assets/img/publication_preview/`
- Recommended: 400x300px
- Reference in bib file as: `preview={filename.png}`

## ⚙️ Configuration Essentials

### Social Media Links
Edit `_data/socials.yml`:
```yaml
- name: Email
  icon: fas fa-envelope
  url: mailto:your.email@usc.edu

- name: Google Scholar
  icon: ai ai-google-scholar
  url: https://scholar.google.com/citations?user=YOUR_ID

- name: GitHub
  icon: fab fa-github
  url: https://github.com/username
```

### Navigation
To hide/show pages, edit front matter in `_pages/` files:
```yaml
nav: false        # Hide from navigation
published: false  # Hide page entirely
```

### About Page
Edit `_pages/about.md`:
- Update `subtitle` for affiliation
- Modify `more_info` for contact details
- Toggle features: `news`, `selected_papers`, `latest_posts`

## 🔧 Common Maintenance Tasks

### Update Bio
Edit `_pages/about.md` content section

### Add News Item
1. Create `_news/announcement_X.md`
2. Add front matter:
```yaml
---
layout: post
date: 2024-06-04
inline: true
related_posts: false
---

Your news content here.
```

### Change Theme Colors
Edit `_sass/_variables.scss`:
```scss
$theme-color: #0366d6;  // Main accent color
```

### Enable/Disable Features
In `_config.yml`:
```yaml
enable_math: true          # LaTeX support
enable_darkmode: true      # Dark/light toggle
enable_google_analytics: false
```

## 🚀 Deployment Workflow

1. **Make changes** to any files
2. **Commit changes** to repository
3. **Wait 5-10 minutes** for GitHub Actions to build
4. **Hard refresh** browser: `Ctrl+Shift+R` / `Cmd+Shift+R`

### Troubleshooting

**Site not updating?**
- Check Actions tab for build errors
- Clear browser cache
- Verify changes were committed to main branch

**Styling broken?**
- Check `baseurl: ""` in `_config.yml`
- Hard refresh browser
- Check for YAML syntax errors

**Math not rendering?**
- Ensure `enable_math: true` in `_config.yml`
- Use `$$` for display math, `$` for inline

## 📁 File Structure Quick Reference

```
your-repo/
├── _pages/
│   ├── about.md          # Homepage content
│   ├── publications.md   # Publications page
│   └── blog.md          # Blog listing
├── _posts/              # Blog posts (YYYY-MM-DD-title.md)
├── _bibliography/
│   └── papers.bib       # Publications database
├── _data/
│   └── socials.yml      # Social media links
├── assets/
│   ├── img/             # Images
│   ├── pdf/             # PDF files
│   └── img/publication_preview/  # Paper thumbnails
└── _config.yml          # Main configuration
```

## 💡 Pro Tips

- **Citation keys**: Use format `lastname2024keyword` for consistency
- **Image optimization**: Compress images before uploading (use TinyPNG)
- **Backup**: Download repository ZIP periodically
- **Preview locally**: Use GitHub's web editor preview for quick checks
- **Mobile check**: Always test on mobile after major changes
- **SEO**: Write good descriptions for all posts and pages

## 🔗 Quick Links

- **Add publication**: Edit `_bibliography/papers.bib`
- **New blog post**: Create in `_posts/` with date format
- **Update bio**: Edit `_pages/about.md`
- **Social links**: Edit `_data/socials.yml`
- **Site settings**: Edit `_config.yml`
- **Upload images**: `assets/img/` folder
- **Check build status**: Repository → Actions tab