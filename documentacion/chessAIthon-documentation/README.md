# Website

This website is built using [Docusaurus](https://docusaurus.io/), a modern static website generator.

## Installation

```bash
yarn
```

## Local Development

```bash
yarn start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

## Build

```bash
yarn build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

## Deployment

Using SSH:

```bash
USE_SSH=true yarn deploy
```

````markdown
# ChessAIthon — Documentation site

This repository contains the documentation site for the ChessAIthon project, built with Docusaurus.

## Quick start

Using npm:

```bash
npm install
npm run start
```

Or using yarn:

```bash
yarn
yarn start
```

The development server runs at http://localhost:3000 and supports live reload while you edit docs and pages.

## Build

Generate a production build:

```bash
npm run build
# or
yarn build
```

The static site will be generated into the `build` directory.

## Deployment

This project is configured to be deployable to GitHub Pages. Example with npm (SSH):

```bash
USE_SSH=true npm run deploy
```

Or with yarn:

```bash
USE_SSH=true yarn deploy
```

If you prefer a different hosting provider, upload the contents of `build` to your static host.

## Contributing

Docs are editable — use the "Edit this page" links in the site header to propose changes via GitHub.

````
