import ccxt

import yaml

from utils import load_config  # Import load_config from utils



def initialize_exchange():

    config = load_config()  # Load the configuration using the utility function



    # Initialize the Phemex exchange

    phemex_futures = ccxt.phemex({

        'apiKey': config['phemex']['api_key'],

        'secret': config['phemex']['api_secret'],

        'urls': {

            'api': {

                'public': config['phemex']['urls']['api']['public'],

                'private': config['phemex']['urls']['api']['private'],

            }

        },

        'enableRateLimit': True,

        'options': {

            'defaultType': 'future'

        }

    })



    return phemex_futures