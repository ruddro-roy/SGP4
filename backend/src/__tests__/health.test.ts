import request from 'supertest';
import express from 'express';

describe('Health Check', () => {
  let app: express.Application;

  beforeAll(() => {
    app = express();
    app.get('/api/health', (req, res) => {
      res.status(200).json({
        status: 'healthy',
        service: 'satellite-tracking-gateway',
        timestamp: new Date().toISOString()
      });
    });
  });

  describe('GET /api/health', () => {
    it('should return 200 with health status', async () => {
      const response = await request(app)
        .get('/api/health')
        .expect(200);

      expect(response.body).toHaveProperty('status', 'healthy');
      expect(response.body).toHaveProperty('service', 'satellite-tracking-gateway');
      expect(response.body).toHaveProperty('timestamp');
    });

    it('should return valid timestamp', async () => {
      const response = await request(app)
        .get('/api/health')
        .expect(200);

      const timestamp = new Date(response.body.timestamp);
      expect(timestamp).toBeInstanceOf(Date);
      expect(timestamp.getTime()).not.toBeNaN();
    });
  });
});