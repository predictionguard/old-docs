import React from 'react'
import { useConfig } from 'nextra-theme-docs'
import { useRouter } from 'next/router'

export default {
  logo: <span>♜ prediction<b>Guard</b></span>,
  logoLink: 'https://docs.predictionguard.com',
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
  useNextSeoProps() {
    return {
      titleTemplate: '%s – Prediction Guard'
    }
  }
}

//export default config