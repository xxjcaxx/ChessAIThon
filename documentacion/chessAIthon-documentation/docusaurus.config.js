// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import {themes as prismThemes} from 'prism-react-renderer';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'ChessAIthon',
  tagline: 'Technical documentation for ChessAIthon — practical chess AI experiments, datasets and models',
  favicon: 'img/favicon.ico',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: 'https://xxjcaxx.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/ChessAIThon/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'xxjcaxx', // Usually your GitHub org/user name.
  projectName: 'ChessAIThon', // Usually your repo name.

  onBrokenLinks: 'throw',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  themes: [
  
    '@docusaurus/theme-mermaid', // ⬅️ AÑADIR ESTA LÍNEA
  ],
  markdown: { 
    mermaid: true, 
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
   
        docs: {
          sidebarPath: './sidebars.js',
          // Edit this to point to the documentation folder in your repo
          editUrl:
            'https://github.com/xxjcaxx/ChessAIThon/tree/master/documentacion/chessAIthon-documentation/',
         
          },
        // Disable the blog for this documentation-focused site
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      colorMode: {
        respectPrefersColorScheme: true,
      },
      navbar: {
        title: 'ChessAIthon',
        logo: {
          alt: 'ChessAIthon Logo',
          src: 'img/logo.png',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Docs',
          },
          // blog removed
          {
            href: 'https://github.com/xxjcaxx/ChessAIThon',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              {
                label: 'Tutorial',
                to: '/docs/intro',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/xxjcaxx/ChessAIThon',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} ChessAIthon. Built with Docusaurus.`,
         logo: {
          alt: 'Financiado por la Unión Europea',
          src: 'img/erasmus_plus_ok.jpg', // Ruta relativa desde /static
         // href: '/', // opcional: a dónde lleva el logo al hacer clic
          width: 800, // puedes ajustar el tamaño
        },
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
    }),
};

export default config;
