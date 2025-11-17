// @ts-check

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.

 @type {import('@docusaurus/plugin-content-docs').SidebarsConfig}
 */
const sidebars = {
  // Manual sidebar for ChessAIthon documentation
  tutorialSidebar: [
    'intro',

    {
      type: 'category',
      label: 'Background',
      collapsed: false,
      items: [
        'chess_computers',
        'architecture_diagrams',
      ],
    },

    {
      type: 'category',
      label: 'Models & Analysis',
      items: [
        'Exploring Chessmaro AI Model',
        'deep_learning',
        'auto_generated_technical_docs',
      ],
    },

    {
      type: 'category',
      label: 'Datasets',
      items: [
        'chess_datasets',
      ],
    },

    {
      type: 'category',
      label: 'Training & Testing',
      items: [
        'training_chessmarro',
        'testing_chessmarro',
      ],
    },

    {
      type: 'category',
      label: 'Deployment',
      items: [
        'deploying_chessmarro',
      ],
    },

    'technical_memory',

   
  ],
};

export default sidebars;
