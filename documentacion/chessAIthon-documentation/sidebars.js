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
    'pedagogical_manual',

    {
      type: 'category',
      label: 'Background',
      collapsed: false,
      items: [
        'chess_computers',
        'chess_datasets',
        'deep_learning',
      ],
    },
    {
      type: 'category',
      label: 'Training & Testing',
      items: [
        'training_chessmarro',
        'Exploring Chessmaro AI Model',

      ],
    },

    {
      type: 'category',
      label: 'Deployment',
      items: [
        'deploying_chessmarro',
        'architecture_diagrams',
      ],
    },

    {
      type: 'category',
      label: 'ChessAIthon Competition',
      items: [
        'chessaithon_competition',
      ],
    },

    {
      type: 'category',
      label: 'Memory',
      items: [
        'technical_memory',
        'auto_generated_technical_docs',
        'slides_marp'
      ],
    },



   
  ],
};

export default sidebars;
