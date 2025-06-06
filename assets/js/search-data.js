// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "About",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "Blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-publications",
          title: "Publications",
          description: "publications in reversed chronological order",
          section: "Navigation",
          handler: () => {
            window.location.href = "/publications/";
          },
        },{id: "post-variational-inference",
        
          title: "Variational Inference",
        
        description: "A note on variational inference and VAEs.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/Variational-Inference/";
          
        },
      },{id: "post-in-praise-of-einsum",
        
          title: "In Praise of Einsum",
        
        description: "A tutorial on einsum, Einstein summation notation.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/In-Praise-of-einsum/";
          
        },
      },{id: "post-map-elites",
        
          title: "MAP-Elites",
        
        description: "An introductory note on the MAP-Elites algorithm.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2022/MAP-Elites/";
          
        },
      },{id: "post-learning-to-score-behaviors",
        
          title: "Learning to Score Behaviors",
        
        description: "An extended summary of &quot;Learning to Score Behaviors for Guided Policy Optimization&quot;.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2022/Learning-to-Score-Behaviors/";
          
        },
      },{id: "post-optimization-primer",
        
          title: "Optimization Primer",
        
        description: "An introduction to (non-convex) optimization.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2022/Optimization-Primer/";
          
        },
      },{id: "post-introduction-to-information-theory",
        
          title: "Introduction to Information Theory",
        
        description: "A brief introduction to information theory, definitions and basic theorems.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2022/Introduction-to-Information-Theory/";
          
        },
      },{id: "post-notes-on-stochastic-processes",
        
          title: "Notes on Stochastic Processes",
        
        description: "The notes I wrote for an undergrad stochastic processes course that I took.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2022/Notes-on-Stochastic-Processes/";
          
        },
      },{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%73%61%65%65%64%68%65%64@%75%73%63.%65%64%75", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/conflictednerd", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/saeed-hedayatian", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=VorwYP0AAAAJ", "_blank");
        },
      },{
        id: 'social-x',
        title: 'X',
        section: 'Socials',
        handler: () => {
          window.open("https://twitter.com/HedayatianSaeed", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
