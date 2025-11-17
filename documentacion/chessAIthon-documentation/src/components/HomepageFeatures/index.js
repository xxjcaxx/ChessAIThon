import clsx from 'clsx';
import Heading from '@theme/Heading';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Overview & Quick Start',
    Svg: require('@site/static/img/inic1.png').default,
    description: (
      <>
        Start here to get a quick overview of ChessAIthon, where the datasets
        live and how to run the examples locally.
      </>
    ),
    href: '/docs/intro',
  },
  {
    title: 'Training Recipes',
    Svg: require('@site/static/img/inici2.svg').default,
    description: (
      <>
        Read the training notes and reproducibility tips for Chessmarro and
        other models used in ChessAIthon.
      </>
    ),
    href: '/docs/training_chessmarro',
  },
  {
    title: 'Deploy & Run',
    Svg: require('@site/static/img/inici3.png').default,
    description: (
      <>
        Examples and recipes to run trained models locally, in containers, or
        deployed to a server for inference.
      </>
    ),
    href: '/docs/deploying_chessmarro',
  },
];

function Feature({Svg, title, description, href}) {
  // Svg can be either a React component (for inline SVGs) or a string URL (PNG/JPG)
  const isStringImage = typeof Svg === 'string';

  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        {isStringImage ? (
          <img src={Svg} className={styles.featureSvg} alt={title} />
        ) : (
          // Svg is a React component
          Svg ? <Svg className={styles.featureSvg} role="img" /> : null
        )}
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
        {href && (
          <div style={{marginTop: '12px'}}>
            <Link className="button button--outline" to={href}>
              Learn more
            </Link>
          </div>
        )}
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
