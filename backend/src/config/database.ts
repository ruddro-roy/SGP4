import mongoose from 'mongoose';
import { logger } from '@/utils/logger';

interface DatabaseConfig {
  uri: string;
  options: mongoose.ConnectOptions;
}

class DatabaseConnection {
  private config: DatabaseConfig;
  private connectionAttempts = 0;
  private maxRetries = 5;
  private retryInterval = 5000; // 5 seconds

  constructor() {
    this.config = {
      uri: process.env.MONGODB_URI || 'mongodb://localhost:27017/satellite-tracker',
      options: {
        maxPoolSize: 10,
        serverSelectionTimeoutMS: 5000,
        socketTimeoutMS: 45000,
        bufferCommands: false,
        bufferMaxEntries: 0,
        retryWrites: true,
        w: 'majority',
        readPreference: 'primaryPreferred',
      },
    };
  }

  public async connect(): Promise<void> {
    try {
      await this.attemptConnection();
      this.setupEventListeners();
      logger.info('Database connected successfully');
    } catch (error) {
      logger.error({ error }, 'Failed to connect to database');
      throw error;
    }
  }

  private async attemptConnection(): Promise<void> {
    while (this.connectionAttempts < this.maxRetries) {
      try {
        await mongoose.connect(this.config.uri, this.config.options);
        return;
      } catch (error) {
        this.connectionAttempts++;
        logger.warn({
          attempt: this.connectionAttempts,
          maxRetries: this.maxRetries,
          error: error instanceof Error ? error.message : 'Unknown error'
        }, 'Database connection attempt failed');

        if (this.connectionAttempts >= this.maxRetries) {
          throw new Error(`Failed to connect to database after ${this.maxRetries} attempts`);
        }

        await this.delay(this.retryInterval);
      }
    }
  }

  private setupEventListeners(): void {
    mongoose.connection.on('connected', () => {
      logger.info('MongoDB connected');
      this.connectionAttempts = 0; // Reset on successful connection
    });

    mongoose.connection.on('error', (error) => {
      logger.error({ error }, 'MongoDB connection error');
    });

    mongoose.connection.on('disconnected', () => {
      logger.warn('MongoDB disconnected');
      // Attempt to reconnect
      this.handleReconnection();
    });

    mongoose.connection.on('reconnected', () => {
      logger.info('MongoDB reconnected');
    });
  }

  private async handleReconnection(): Promise<void> {
    if (process.env.SELF_HEALING_ENABLED === 'true') {
      logger.info('Attempting database reconnection (self-healing enabled)');
      try {
        await this.connect();
      } catch (error) {
        logger.error({ error }, 'Self-healing database reconnection failed');
      }
    }
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  public async disconnect(): Promise<void> {
    try {
      await mongoose.disconnect();
      logger.info('Database disconnected gracefully');
    } catch (error) {
      logger.error({ error }, 'Error disconnecting from database');
    }
  }

  public getConnectionState(): string {
    const states = {
      0: 'disconnected',
      1: 'connected',
      2: 'connecting',
      3: 'disconnecting',
    };
    return states[mongoose.connection.readyState as keyof typeof states] || 'unknown';
  }
}

// Export singleton instance
const databaseConnection = new DatabaseConnection();

export const connectDatabase = (): Promise<void> => databaseConnection.connect();
export const disconnectDatabase = (): Promise<void> => databaseConnection.disconnect();
export const getDatabaseState = (): string => databaseConnection.getConnectionState();

// Health check function
export const isDatabaseHealthy = (): boolean => {
  return mongoose.connection.readyState === 1; // connected
};