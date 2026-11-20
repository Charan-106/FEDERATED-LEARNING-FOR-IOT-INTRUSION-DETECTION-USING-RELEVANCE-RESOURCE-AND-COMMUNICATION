// ARC-FL Backend - Express Application Entry Point
// Sets up middleware, mounts routes, starts the HTTP server

require('dotenv').config();

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
const rateLimit = require('express-rate-limit');
const path = require('path');
const fs = require('fs');

const config = require('./src/config');
const prisma = require('./src/config/database');
const errorHandler = require('./src/middleware/errorHandler');
const { initModelReferences } = require('./src/services/modelLifecycleService');
const { ensureDefaultArcConfig } = require('./src/controllers/arcController');

// Route modules
const authRoutes = require('./src/routes/auth');
const clientRoutes = require('./src/routes/client');
const adminRoutes = require('./src/routes/admin');
const arcRoutes = require('./src/routes/arc');
const aggregationRoutes = require('./src/routes/aggregation');
const modelRoutes = require('./src/routes/model');

const app = express();

// ============================================================
// Security Middleware
// ============================================================
app.use(helmet({ contentSecurityPolicy: false })); // Sets secure HTTP headers
app.use(cors({
  origin: (origin, callback) => {
    // Allow non-browser requests (e.g. curl, server-to-server, edge client-agent scripts)
    if (!origin) return callback(null, true);

    // Allow localhost, 127.0.0.1, private LAN subnets (10.x, 192.168.x, 172.16-31.x), and ngrok tunnels
    const isAllowed =
      /^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(origin) ||
      /^http:\/\/(10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(1[6-9]|2\d|3[0-1])\.\d+\.\d+)(:\d+)?$/.test(origin) ||
      /^https:\/\/.*\.ngrok(-free)?\.(app|dev|io)$/.test(origin) ||
      config.nodeEnv === 'development'; // Always permit LAN dev connections in development

    if (isAllowed) {
      return callback(null, true);
    }
    return callback(new Error(`CORS origin not allowed: ${origin}`));
  },
  credentials: true,
}));

// Global rate limiter: 300 requests per 15 minutes per IP
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 300,
  message: { error: 'Too many requests', message: 'Please slow down and try again later.' },
  standardHeaders: true,
  legacyHeaders: false,
});
app.use(limiter);

// ============================================================
// Request Parsing
// ============================================================
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// HTTP request logging (dev: colorized, prod: combined)
app.use(morgan(config.nodeEnv === 'development' ? 'dev' : 'combined'));

// ============================================================
// Health Check (no auth required)
// ============================================================
app.get('/health', async (req, res) => {
  let dbStatus = 'unknown';
  try {
    await prisma.$queryRaw`SELECT 1`;
    dbStatus = 'connected';
  } catch {
    dbStatus = 'disconnected';
  }

  res.json({
    status: 'ok',
    service: 'arc-fl-backend',
    version: '1.0.0',
    timestamp: new Date().toISOString(),
    database: dbStatus,
  });
});

app.get('/api/health', async (req, res) => {
  let dbStatus = 'unknown';
  try {
    await prisma.$queryRaw`SELECT 1`;
    dbStatus = 'connected';
  } catch {
    dbStatus = 'disconnected';
  }

  res.json({
    status: 'ok',
    service: 'arc-fl-backend',
    version: '1.0.0',
    timestamp: new Date().toISOString(),
    database: dbStatus,
  });
});

// ============================================================
// API Routes
// ============================================================
app.use('/api/auth', authRoutes);
app.use('/api/client', clientRoutes);
app.use('/api/admin', adminRoutes);
app.use('/api/arc', arcRoutes);
app.use('/api/aggregation', aggregationRoutes);
app.use('/api/models', modelRoutes);

// ============================================================
// Serve Frontend Static Distribution
// ============================================================
const frontendDist = path.join(__dirname, '../frontend/dist');
if (fs.existsSync(frontendDist)) {
  app.use(express.static(frontendDist));
  app.get('*', (req, res, next) => {
    if (req.path.startsWith('/api') || req.path === '/health') return next();
    res.sendFile(path.join(frontendDist, 'index.html'));
  });
}

// 404 handler for unknown routes
app.use((req, res) => {
  res.status(404).json({
    error: 'NotFound',
    message: `Route ${req.method} ${req.path} not found`,
    timestamp: new Date().toISOString(),
  });
});

// ============================================================
// Central Error Handler (must be last)
// ============================================================
app.use(errorHandler);

// ============================================================
// Start Server
// ============================================================
const PORT = config.port;

async function start() {
  try {
    // Test database connection before accepting traffic
    await prisma.$connect();
    console.log('[DB] PostgreSQL connected successfully');

    // Initialize & verify authoritative model references
    try {
      await initModelReferences();
      console.log('[ModelLifecycle] Authoritative model references synchronized');
    } catch (modelInitErr) {
      console.warn('[ModelLifecycle] Warning during model references init:', modelInitErr.message);
    }

    // Initialize & verify authoritative ARC configuration
    try {
      await ensureDefaultArcConfig();
      console.log('[ARC] Authoritative ARC configuration synchronized');
    } catch (arcInitErr) {
      console.warn('[ARC] Warning during ARC config init:', arcInitErr.message);
    }

    app.listen(PORT, "0.0.0.0" ,() => {
      console.log("Backend running on port 5000");
      console.log(`[Server] ARC-FL Backend running on http://localhost:${PORT}`);
      console.log(`[Server] Environment: ${config.nodeEnv}`);
      console.log(`[Server] ML Service: ${config.mlService.url}`);
    });
  } catch (err) {
    console.error('[Server] Failed to start:', err.message);
    process.exit(1);
  }
}

// Handle unhandled promise rejections and uncaught exceptions gracefully
process.on('unhandledRejection', (reason) => {
  console.error('[Process] Unhandled rejection:', reason);
});

process.on('uncaughtException', (err) => {
  console.error('[Process] Uncaught exception:', err);
});

start();
