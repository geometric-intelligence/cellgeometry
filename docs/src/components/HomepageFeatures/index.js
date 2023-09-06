import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Easy to Use',
    Svg: require('@site/static/img/Asset1.svg').default,
    description: (
      <>
        Cellgeometry was designed to be easy to use, no local installation necessary. Everything is stored and computed online in the web browser.
      </>
    ),
  },
  {
    title: 'Focus on What Matters',
    Svg: require('@site/static/img/Asset1.svg').default,
    description: (
      <>
        CellGeometry lets you focus on analyzing cell shapes, and we will do the chores. Go
        ahead and move your upload some data to the <code>app</code>.
      </>
    ),
  },
  {
    title: 'Powered by Python',
    Svg: require('@site/static/img/Asset1.svg').default,
    description: (
      <>
        Extend or customize CellGeometry to suit your needs. Feel free to file PRs and share your methods on our platform for everyone to use.
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
