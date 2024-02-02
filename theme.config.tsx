import React from 'react'
import { useConfig } from 'nextra-theme-docs'
import { useRouter } from 'next/router'
import Script from 'next/script'

export default {
  logo: <span>♜ prediction<b>Guard</b></span>,
  logoLink: 'https://www.predictionguard.com/',
  primaryHue: 136,
  project: {
    link: 'https://github.com/predictionguard/docs',
  },
  // chat: {
  //   link: 'https://discord.com',
  // },
  docsRepositoryBase: 'https://github.com/predictionguard/docs',
  footer: {
    text: '♜ Prediction Guard docs',
  },
  faviconGlyph: '♜',
  // Add in static head tags.
  // See https://nextjs.org/docs/api-reference/next/head
  head: (
    <>
      <Script id="clarityinsert"
        strategy="afterInteractive"
        dangerouslySetInnerHTML={{
          __html: `
          (function(c,l,a,r,i,t,y){
            c[a]=c[a]||function(){(c[a].q=c[a].q||[]).push(arguments)};
            t=l.createElement(r);t.async=1;t.src="https://www.clarity.ms/tag/"+i;
            y=l.getElementsByTagName(r)[0];y.parentNode.insertBefore(t,y);
        })(window, document, "clarity", "script", 'gpm8utc3xm');`,
        }}
      />
    </>
  ),

  useNextSeoProps() {
    return {
      titleTemplate: '%s – Prediction Guard'
    }
  }
}

//export default config
