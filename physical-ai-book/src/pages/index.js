import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <h1 className="hero__title">
          Physical AI & Humanoid Robotics Book
        </h1>
        <p className="hero__subtitle">
          The Future of Intelligent Systems
        </p>
        <div className={styles.buttons}>
          <div className={styles.robotContainer}>
            <div className={styles.digitalBrain}></div>
            <div className={styles.robotSilhouette}>
              <div className={clsx(styles.robotLimb, styles.robotArmLeft)}></div>
              <div className={clsx(styles.robotLimb, styles.robotArmRight)}></div>
              <div className={clsx(styles.robotLimb, styles.robotLegLeft)}></div>
              <div className={clsx(styles.robotLimb, styles.robotLegRight)}></div>
            </div>
          </div>
          <div className={styles.buttonGroup}>
            <Link
              className="button button--secondary button--lg"
              to="/docs/intro">
              Start Reading
            </Link>
            <Link
              className="button button--primary button--lg"
              to="/docs/module-1-ros2">
              Explore Chapters
            </Link>
          </div>
        </div>
      </div>
      
      {/* Floating particles */}
      <div className={styles.particle}></div>
      <div className={clsx(styles.particle, styles.particle2)}></div>
      <div className={clsx(styles.particle, styles.particle3)}></div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="Description will go into a meta tag in <head />">
      <HomepageHeader />
      <main>
        {/* Additional sections with futuristic styling */}
        <section className={styles.features}>
          <div className="container">
            <div className="row">
              <div className={clsx('col col--4', styles.featureCard)}>
                <h3>🤖 Advanced AI Integration</h3>
                <p>Learn how to integrate artificial intelligence systems with physical robotic platforms for unprecedented capabilities.</p>
              </div>
              <div className={clsx('col col--4', styles.featureCard)}>
                <h3>🌐 Digital Twins</h3>
                <p>Master simulation environments that mirror real-world physics for safe robotics development and testing.</p>
              </div>
              <div className={clsx('col col--4', styles.featureCard)}>
                <h3>🚀 Real Robots</h3>
                <p>Transition from simulation to controlling actual humanoid robots with confidence and precision.</p>
              </div>
            </div>
          </div>
        </section>
        
        {/* Hologram-style section */}
        <section className={styles.hologramSection}>
          <div className="container">
            <h2>Why Physical AI?</h2>
            <p>The future of artificial intelligence lies not just in digital systems, but in embodied agents that can interact with and manipulate the physical world. This book bridges the gap between AI algorithms and real-world applications.</p>
          </div>
        </section>
      </main>
    </Layout>
  );
}