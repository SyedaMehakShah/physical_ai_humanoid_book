import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/physical-ai-book/__docusaurus/debug',
    component: ComponentCreator('/physical-ai-book/__docusaurus/debug', '12f'),
    exact: true
  },
  {
    path: '/physical-ai-book/__docusaurus/debug/config',
    component: ComponentCreator('/physical-ai-book/__docusaurus/debug/config', '4d3'),
    exact: true
  },
  {
    path: '/physical-ai-book/__docusaurus/debug/content',
    component: ComponentCreator('/physical-ai-book/__docusaurus/debug/content', 'a5b'),
    exact: true
  },
  {
    path: '/physical-ai-book/__docusaurus/debug/globalData',
    component: ComponentCreator('/physical-ai-book/__docusaurus/debug/globalData', 'abe'),
    exact: true
  },
  {
    path: '/physical-ai-book/__docusaurus/debug/metadata',
    component: ComponentCreator('/physical-ai-book/__docusaurus/debug/metadata', '587'),
    exact: true
  },
  {
    path: '/physical-ai-book/__docusaurus/debug/registry',
    component: ComponentCreator('/physical-ai-book/__docusaurus/debug/registry', '2ef'),
    exact: true
  },
  {
    path: '/physical-ai-book/__docusaurus/debug/routes',
    component: ComponentCreator('/physical-ai-book/__docusaurus/debug/routes', '1a5'),
    exact: true
  },
  {
    path: '/physical-ai-book/docs',
    component: ComponentCreator('/physical-ai-book/docs', 'bb3'),
    routes: [
      {
        path: '/physical-ai-book/docs',
        component: ComponentCreator('/physical-ai-book/docs', '7b9'),
        routes: [
          {
            path: '/physical-ai-book/docs',
            component: ComponentCreator('/physical-ai-book/docs', '754'),
            routes: [
              {
                path: '/physical-ai-book/docs/intro',
                component: ComponentCreator('/physical-ai-book/docs/intro', '050'),
                exact: true
              },
              {
                path: '/physical-ai-book/docs/labs/lab-template',
                component: ComponentCreator('/physical-ai-book/docs/labs/lab-template', 'eef'),
                exact: true
              },
              {
                path: '/physical-ai-book/docs/module-1-ros2/',
                component: ComponentCreator('/physical-ai-book/docs/module-1-ros2/', '50e'),
                exact: true
              },
              {
                path: '/physical-ai-book/docs/module-1-ros2/nodes',
                component: ComponentCreator('/physical-ai-book/docs/module-1-ros2/nodes', 'cfb'),
                exact: true
              },
              {
                path: '/physical-ai-book/docs/module-1-ros2/rclpy',
                component: ComponentCreator('/physical-ai-book/docs/module-1-ros2/rclpy', '45d'),
                exact: true
              },
              {
                path: '/physical-ai-book/docs/module-1-ros2/services',
                component: ComponentCreator('/physical-ai-book/docs/module-1-ros2/services', '8e3'),
                exact: true
              },
              {
                path: '/physical-ai-book/docs/module-1-ros2/topics',
                component: ComponentCreator('/physical-ai-book/docs/module-1-ros2/topics', '97b'),
                exact: true
              },
              {
                path: '/physical-ai-book/docs/module-1-ros2/urdf',
                component: ComponentCreator('/physical-ai-book/docs/module-1-ros2/urdf', 'a5e'),
                exact: true
              },
              {
                path: '/physical-ai-book/docs/module-2-digital-twin/',
                component: ComponentCreator('/physical-ai-book/docs/module-2-digital-twin/', '790'),
                exact: true
              },
              {
                path: '/physical-ai-book/docs/module-2-digital-twin/gazebo',
                component: ComponentCreator('/physical-ai-book/docs/module-2-digital-twin/gazebo', 'b65'),
                exact: true
              },
              {
                path: '/physical-ai-book/docs/module-2-digital-twin/physics',
                component: ComponentCreator('/physical-ai-book/docs/module-2-digital-twin/physics', '11c'),
                exact: true
              },
              {
                path: '/physical-ai-book/docs/module-2-digital-twin/sensors',
                component: ComponentCreator('/physical-ai-book/docs/module-2-digital-twin/sensors', 'cfe'),
                exact: true
              },
              {
                path: '/physical-ai-book/docs/module-2-digital-twin/unity',
                component: ComponentCreator('/physical-ai-book/docs/module-2-digital-twin/unity', '728'),
                exact: true
              },
              {
                path: '/physical-ai-book/docs/module-3-isaac/',
                component: ComponentCreator('/physical-ai-book/docs/module-3-isaac/', '93f'),
                exact: true
              },
              {
                path: '/physical-ai-book/docs/module-3-isaac/ai-integration',
                component: ComponentCreator('/physical-ai-book/docs/module-3-isaac/ai-integration', 'ad1'),
                exact: true
              },
              {
                path: '/physical-ai-book/docs/module-3-isaac/isaac-sim',
                component: ComponentCreator('/physical-ai-book/docs/module-3-isaac/isaac-sim', 'e07'),
                exact: true
              },
              {
                path: '/physical-ai-book/docs/module-4-real-robots/',
                component: ComponentCreator('/physical-ai-book/docs/module-4-real-robots/', '945'),
                exact: true
              },
              {
                path: '/physical-ai-book/docs/module-4-real-robots/deployment',
                component: ComponentCreator('/physical-ai-book/docs/module-4-real-robots/deployment', '18e'),
                exact: true
              },
              {
                path: '/physical-ai-book/docs/module-4-real-robots/robot-hardware',
                component: ComponentCreator('/physical-ai-book/docs/module-4-real-robots/robot-hardware', '3e8'),
                exact: true
              },
              {
                path: '/physical-ai-book/docs/module-4-real-robots/safety',
                component: ComponentCreator('/physical-ai-book/docs/module-4-real-robots/safety', '20f'),
                exact: true
              },
              {
                path: '/physical-ai-book/docs/module-5-hri-safety/',
                component: ComponentCreator('/physical-ai-book/docs/module-5-hri-safety/', '05f'),
                exact: true
              },
              {
                path: '/physical-ai-book/docs/module-5-hri-safety/ethics',
                component: ComponentCreator('/physical-ai-book/docs/module-5-hri-safety/ethics', 'ca7'),
                exact: true
              },
              {
                path: '/physical-ai-book/docs/module-5-hri-safety/interaction-design',
                component: ComponentCreator('/physical-ai-book/docs/module-5-hri-safety/interaction-design', '3be'),
                exact: true
              },
              {
                path: '/physical-ai-book/docs/module-5-hri-safety/safety-protocols',
                component: ComponentCreator('/physical-ai-book/docs/module-5-hri-safety/safety-protocols', '0b5'),
                exact: true
              },
              {
                path: '/physical-ai-book/docs/resources/faq',
                component: ComponentCreator('/physical-ai-book/docs/resources/faq', 'daf'),
                exact: true
              },
              {
                path: '/physical-ai-book/docs/resources/glossary',
                component: ComponentCreator('/physical-ai-book/docs/resources/glossary', '0af'),
                exact: true
              },
              {
                path: '/physical-ai-book/docs/resources/lab-template',
                component: ComponentCreator('/physical-ai-book/docs/resources/lab-template', 'c6b'),
                exact: true
              },
              {
                path: '/physical-ai-book/docs/resources/references',
                component: ComponentCreator('/physical-ai-book/docs/resources/references', '6c1'),
                exact: true
              },
              {
                path: '/physical-ai-book/docs/resources/spec-driven-workflow',
                component: ComponentCreator('/physical-ai-book/docs/resources/spec-driven-workflow', '7df'),
                exact: true
              },
              {
                path: '/physical-ai-book/docs/resources/tools',
                component: ComponentCreator('/physical-ai-book/docs/resources/tools', '235'),
                exact: true
              },
              {
                path: '/physical-ai-book/docs/resources/tools-overview',
                component: ComponentCreator('/physical-ai-book/docs/resources/tools-overview', '228'),
                exact: true
              },
              {
                path: '/physical-ai-book/docs/resources/troubleshooting',
                component: ComponentCreator('/physical-ai-book/docs/resources/troubleshooting', '64a'),
                exact: true
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/physical-ai-book/',
    component: ComponentCreator('/physical-ai-book/', '003'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
