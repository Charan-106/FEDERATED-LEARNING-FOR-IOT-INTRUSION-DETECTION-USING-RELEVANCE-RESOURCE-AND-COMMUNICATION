// ML Service Client - HTTP client for communicating with the Python FastAPI service
// All calls include the shared API key for authentication

const axios = require('axios');
const config = require('../config');

// Create an axios instance pre-configured for the ML service
const mlAxios = axios.create({
  baseURL: config.mlService.url,
  timeout: 300_000, // 5 minutes (training/aggregation can take time)
  headers: {
    'X-API-Key': config.mlService.apiKey,
    'Content-Type': 'application/json',
  },
});

// Log ML service calls in development
mlAxios.interceptors.request.use((reqConfig) => {
  if (config.nodeEnv === 'development') {
    console.log(`[MLService] ${reqConfig.method?.toUpperCase()} ${reqConfig.url}`);
  }
  return reqConfig;
});

mlAxios.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const msg = error.response?.data?.detail || error.response?.data?.message || error.message;
    console.error(`[MLService] Error: ${msg}`);
    throw new Error(msg);
  }
);

/**
 * calculateARC - Sends client telemetry and submission metadata to the ML service
 * to compute the ARC score (Relevance, Resource, Communication components).
 */
async function calculateARC(payload) {
  const arcPayload = {
    submission_id: String(payload.submissionId || payload.submission_id || 'sub-001'),
    client_id: String(payload.clientId || payload.client_id || 'client-001'),
    round_id: String(payload.roundId || payload.round_id || 'round-1'),
    local_accuracy: parseFloat(payload.localAccuracy || payload.local_accuracy || 0),
    local_loss: parseFloat(payload.localLoss || payload.local_loss || 0),
    samples_trained: parseInt(payload.samplesTrained || payload.samples_trained || 0),
    arc_config: payload.arcConfig || payload.arc_config || null,
    quality_metrics: {
      quality_score: payload.quality_metrics?.quality_score !== undefined ? payload.quality_metrics.quality_score : (payload.telemetry?.quality_score ?? null),
      feature_noise_score: payload.quality_metrics?.feature_noise_score ?? payload.telemetry?.feature_noise_score ?? null,
      label_agreement_score: payload.quality_metrics?.label_agreement_score ?? payload.quality_metrics?.label_consistency_score ?? payload.telemetry?.label_agreement_score ?? payload.telemetry?.label_consistency_score ?? null,
      label_consistency_score: payload.quality_metrics?.label_agreement_score ?? payload.quality_metrics?.label_consistency_score ?? payload.telemetry?.label_agreement_score ?? payload.telemetry?.label_consistency_score ?? null,
      missing_ratio: payload.quality_metrics?.missing_ratio ?? payload.telemetry?.missing_ratio ?? 0.0,
      inf_ratio: payload.quality_metrics?.inf_ratio ?? payload.telemetry?.inf_ratio ?? 0.0,
      preprocessing_success: payload.quality_metrics?.preprocessing_success ?? payload.telemetry?.preprocessing_success ?? 1.0,
      missing_cells: payload.quality_metrics?.missing_cells ?? payload.telemetry?.missing_cells ?? 0,
      total_cells: payload.quality_metrics?.total_cells ?? payload.telemetry?.total_cells ?? 0,
    },
    diversity_metrics: {
      diversity_score: payload.diversity_metrics?.diversity_score !== undefined ? payload.diversity_metrics.diversity_score : (payload.telemetry?.diversity_score ?? null),
      class_count: payload.diversity_metrics?.class_count ?? payload.telemetry?.class_count ?? 1,
      entropy: payload.diversity_metrics?.entropy ?? payload.telemetry?.entropy ?? 0.0,
      minority_class_coverage: payload.diversity_metrics?.minority_class_coverage ?? payload.telemetry?.minority_class_coverage ?? 0.0,
    },
    resource_metrics: {
      available_ram_gb: (payload.resource_metrics?.available_ram_gb !== undefined ? payload.resource_metrics.available_ram_gb : payload.telemetry?.available_ram_gb) ?? null,
      total_ram_gb: (payload.resource_metrics?.total_ram_gb !== undefined ? payload.resource_metrics.total_ram_gb : payload.telemetry?.total_ram_gb) ?? null,
      cpu_idle_pct: (payload.resource_metrics?.cpu_idle_pct !== undefined ? payload.resource_metrics.cpu_idle_pct : payload.telemetry?.cpu_idle_pct) ?? null,
      throughput_samples_per_sec: (payload.resource_metrics?.throughput_samples_per_sec !== undefined ? payload.resource_metrics.throughput_samples_per_sec : payload.telemetry?.throughput_samples_per_sec) ?? null,
    },
    comm_metrics: {
      bandwidth_mbps: (payload.comm_metrics?.bandwidth_mbps !== undefined ? payload.comm_metrics.bandwidth_mbps : payload.telemetry?.bandwidth_mbps) ?? null,
      latency_ms: (payload.comm_metrics?.latency_ms !== undefined ? payload.comm_metrics.latency_ms : payload.telemetry?.latency_ms) ?? null,
      reliability_pct: (payload.comm_metrics?.reliability_pct !== undefined ? payload.comm_metrics.reliability_pct : payload.telemetry?.reliability_pct) ?? null,
    },
  };
  return mlAxios.post('/arc/calculate', arcPayload);
}

/**
 * aggregateModels - Triggers federated averaging on the Python side.
 * Python service loads each client .pt file, validates architecture compatibility,
 * performs weighted averaging, and returns the candidate model path.
 */
async function aggregateModels(payload) {
  return mlAxios.post('/aggregation/merge', payload);
}

/**
 * evaluateModel - Runs the candidate model against the validation dataset.
 * Returns accuracy, F1, precision, recall, loss metrics.
 */
async function evaluateModel(candidatePath) {
  return mlAxios.post('/evaluation/validate', { model_path: candidatePath });
}

/**
 * trainLocal - Triggers local edge training on a client's dataset partition.
 * Used for simulation; in production clients train locally and upload updates.
 */
async function trainLocal(payload) {
  return mlAxios.post('/training/start', payload);
}

/**
 * streamLocalTraining - Streams real-time PyTorch training events directly from the edge trainer.
 * Yields epoch progress, batch indices, batch loss, and elapsed seconds.
 */
async function streamLocalTraining(payload, onEvent) {
  const http = require('http');
  const { URL } = require('url');

  const mlUrl = new URL(`${config.mlService.url}/training/stream`);
  const bodyData = JSON.stringify(payload);

  return new Promise((resolve, reject) => {
    const req = http.request(
      {
        hostname: mlUrl.hostname,
        port: mlUrl.port,
        path: mlUrl.pathname,
        method: 'POST',
        headers: {
          'X-API-Key': config.mlService.apiKey,
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(bodyData),
        },
        timeout: 300_000,
      },
      (res) => {
        if (res.statusCode !== 200) {
          let errData = '';
          res.on('data', (d) => (errData += d));
          res.on('end', () => reject(new Error(`ML Service Stream returned ${res.statusCode}: ${errData}`)));
          return;
        }

        let buffer = '';
        let finalResult = null;

        res.on('data', (chunk) => {
          buffer += chunk.toString();
          const lines = buffer.split('\n');
          buffer = lines.pop();

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const event = JSON.parse(line.substring(6));
                if (event.event === 'RESULT' || event.stage === 'RESULT') {
                  finalResult = event.data;
                } else if (event.event === 'ERROR' || event.stage === 'ERROR') {
                  reject(new Error(event.error || 'ML Service training stream error'));
                } else if (onEvent) {
                  onEvent(event);
                }
              } catch (e) {
                // Ignore SSE heartbeat parse issues
              }
            }
          }
        });

        res.on('end', () => {
          if (finalResult) {
            resolve(finalResult);
          } else {
            reject(new Error('ML Service training stream completed without final result.'));
          }
        });
      }
    );

    req.on('error', (err) => reject(err));
    req.write(bodyData);
    req.end();
  });
}

/**
 * getHealth - Checks if the Python ML service is healthy.
 */
async function getHealth() {
  return mlAxios.get('/health');
}

module.exports = { calculateARC, aggregateModels, evaluateModel, trainLocal, streamLocalTraining, getHealth };
