import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import rateLimit from 'express-rate-limit';
import { config } from 'dotenv';
import pino from 'pino';

// Load environment variables
config();

// Create logger
const logger = pino({
  transport: {
    target: 'pino-pretty',
    options: {
      colorize: true,
      translateTime: 'HH:MM:ss Z',
      ignore: 'pid,hostname',
    },
  },
});

// Create Express app
const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(helmet());
app.use(cors({
  origin: process.env.CORS_ORIGIN || '*',
  credentials: true,
}));
app.use(compression());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP, please try again later.',
});
app.use('/api/', limiter);

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({
    status: 'healthy',
    service: 'satellite-tracking-gateway',
    version: process.env.npm_package_version || '1.0.0',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    environment: process.env.NODE_ENV || 'development',
  });
});

// Root endpoint
app.get('/', (req, res) => {
  res.json({
    service: 'Satellite Tracking Gateway',
    version: '1.0.0',
    endpoints: {
      health: '/api/health',
      satellites: '/api/satellites',
      predict: '/api/predict',
      threats: '/api/threats',
    },
  });
});

// Placeholder routes
app.get('/api/satellites', async (req, res) => {
  try {
    // TODO: Implement satellite listing
    res.json({
      satellites: [],
      total: 0,
      page: 1,
      limit: 20,
    });
  } catch (error) {
    logger.error('Error fetching satellites:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.post('/api/predict', async (req, res) => {
  try {
    const { satellite_id, start_time, end_time } = req.body;
    
    if (!satellite_id || !start_time || !end_time) {
      return res.status(400).json({ 
        error: 'Missing required parameters: satellite_id, start_time, end_time' 
      });
    }
    
    // TODO: Implement orbit prediction
    res.json({
      satellite_id,
      predictions: [],
      start_time,
      end_time,
    });
  } catch (error) {
    logger.error('Error predicting orbit:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.get('/api/threats', async (req, res) => {
  try {
    // TODO: Implement threat analysis endpoint
    res.json({
      threats: [],
      analysis_time: new Date().toISOString(),
      total_satellites_analyzed: 0,
    });
  } catch (error) {
    logger.error('Error fetching threats:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Error handling middleware
app.use((err: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
  logger.error('Unhandled error:', err);
  res.status(err.status || 500).json({
    error: err.message || 'Internal server error',
    ...(process.env.NODE_ENV === 'development' && { stack: err.stack }),
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({ error: 'Not found' });
});

// Start server
const server = app.listen(PORT, () => {
  logger.info(`🚀 Satellite Tracking Gateway running on port ${PORT}`);
  logger.info(`Environment: ${process.env.NODE_ENV || 'development'}`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  logger.info('SIGTERM signal received: closing HTTP server');
  server.close(() => {
    logger.info('HTTP server closed');
    process.exit(0);
  });
});

process.on('SIGINT', () => {
  logger.info('SIGINT signal received: closing HTTP server');
  server.close(() => {
    logger.info('HTTP server closed');
    process.exit(0);
  });
});

export default app;