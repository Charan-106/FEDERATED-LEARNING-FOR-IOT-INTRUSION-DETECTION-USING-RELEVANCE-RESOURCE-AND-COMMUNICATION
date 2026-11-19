// Central configuration loader for ARC-FL Backend
// All environment variables are loaded here with defaults and validation

const path = require('path');
require('dotenv').config({ path: path.resolve(__dirname, '../../.env') });

const storageRoot = process.env.STORAGE_ROOT 
  ? path.resolve(process.env.STORAGE_ROOT) 
  : path.resolve(__dirname, '../../../storage');

const config = {
  port: parseInt(process.env.PORT || '5000', 10),
  nodeEnv: process.env.NODE_ENV || 'development',

  jwt: {
    secret: process.env.JWT_SECRET || 'fallback-secret-change-in-production',
    expiresIn: process.env.JWT_EXPIRES_IN || '7d',
  },

  database: {
    url: process.env.DATABASE_URL,
  },

  mlService: {
    url: process.env.ML_SERVICE_URL || 'http://127.0.0.1:8000',
    apiKey: process.env.ML_SERVICE_API_KEY || '',
  },

  storage: {
    root: storageRoot,
    clientUpdates: path.join(storageRoot, 'client-updates'),
    globalModels: path.join(storageRoot, 'global-models'),
    candidates: path.join(storageRoot, 'candidates'),
  },

  upload: {
    maxSizeMb: 100,
  },
};

// Validate required configuration at startup
function validateConfig() {
  const required = ['jwt.secret', 'database.url', 'mlService.apiKey'];
  for (const key of required) {
    const val = key.split('.').reduce((o, k) => (o || {})[k], config);
    if (!val) {
      console.warn(`[Config] WARNING: Missing required config: ${key}`);
    }
  }
}

validateConfig();

module.exports = config;
