// Client Controller - Dashboard, model download, and submission endpoints
// Handles all CLIENT role operations in the federated learning workflow

const path = require('path');
const fs = require('fs');
const crypto = require('crypto');
const prisma = require('../config/database');
const config = require('../config');
const mlServiceClient = require('../services/mlServiceClient');
const { broadcastTrainingProgress, broadcastSubmissionRegistered } = require('./trainingMonitorController');
const { resolveEligibleRoundForClient, getActiveRoundsSummaryForClient } = require('../services/roundLifecycleService');

/**
 * GET /api/client/dashboard
 * Returns overview data for the client's dashboard.
 */
async function getDashboard(req, res, next) {
  try {
    const clientProfile = req.user.clientProfile;
    if (!clientProfile) {
      return res.status(400).json({ error: 'No client profile found' });
    }

    const [submissions, resolvedRoundInfo, activeModel, arcScores, activeRoundsSummary] = await Promise.all([
      prisma.clientSubmission.findMany({
        where: { clientId: clientProfile.clientId },
        orderBy: { submittedAt: 'desc' },
        take: 5,
        include: { round: true },
      }),
      resolveEligibleRoundForClient(clientProfile.clientId),
      prisma.globalModel.findFirst({ where: { isActive: true } }),
      prisma.aRCScore.findMany({
        where: { clientId: clientProfile.clientId },
        orderBy: { calculatedAt: 'desc' },
        take: 5,
      }),
      getActiveRoundsSummaryForClient(clientProfile.clientId),
    ]);

    const totalSubmissions = await prisma.clientSubmission.count({
      where: { clientId: clientProfile.clientId },
    });

    const avgArcScore = arcScores.length > 0
      ? arcScores.reduce((sum, s) => sum + s.overallArcScore, 0) / arcScores.length
      : null;

    res.json({
      clientProfile,
      totalSubmissions,
      recentSubmissions: submissions,
      latestRound: resolvedRoundInfo.round,
      clientEligibility: {
        isTargeted: resolvedRoundInfo.isTargeted,
        hasSubmitted: resolvedRoundInfo.hasSubmitted,
        canTrain: resolvedRoundInfo.isTargeted && !resolvedRoundInfo.hasSubmitted && resolvedRoundInfo.round?.status === 'ACTIVE',
        reason: resolvedRoundInfo.reason,
      },
      activeRounds: activeRoundsSummary,
      activeModel: activeModel
        ? { id: activeModel.id, version: activeModel.version, name: activeModel.name, accuracy: activeModel.baselineAccuracy }
        : null,
      arcScoreSummary: { average: avgArcScore, recent: arcScores },
    });
  } catch (err) {
    next(err);
  }
}

/**
 * GET /api/client/model
 * Returns active global model metadata (not the binary).
 */
async function getGlobalModel(req, res, next) {
  try {
    const model = await prisma.globalModel.findFirst({
      where: { isActive: true },
    });

    if (!model) {
      return res.status(404).json({ error: 'No active global model found' });
    }

    res.json({ model });
  } catch (err) {
    next(err);
  }
}

/**
 * GET /api/client/model/download
 * Streams the active global model's best.pt file to the client.
 */
async function downloadGlobalModel(req, res, next) {
  try {
    let model = null;
    const { roundId, modelId, version } = req.query;

    if (roundId) {
      const round = await prisma.trainingRound.findUnique({
        where: { id: roundId },
        include: { baseModel: true },
      });
      if (round && round.baseModel) {
        model = round.baseModel;
      }
    } else if (modelId) {
      model = await prisma.globalModel.findUnique({ where: { id: modelId } });
    } else if (version) {
      model = await prisma.globalModel.findUnique({ where: { version } });
    }

    if (!model) {
      model = await prisma.globalModel.findFirst({
        where: { isActive: true },
      });
    }

    if (!model) {
      return res.status(404).json({ error: 'No active global model found' });
    }

    // Build path to best.pt inside the model's storage directory
    const modelFilePath = path.join(config.storage.root, model.storagePath, 'best.pt');

    if (!fs.existsSync(modelFilePath)) {
      return res.status(404).json({
        error: 'Model file not found on disk',
        path: modelFilePath,
      });
    }

    // Log the download as an audit event
    await prisma.auditLog.create({
      data: {
        userId: req.user.id,
        action: 'DOWNLOAD_MODEL',
        resourceType: 'GlobalModel',
        resourceId: model.id,
        detailsJson: { version: model.version },
        ipAddress: req.ip,
      },
    });

    res.setHeader('Content-Disposition', `attachment; filename="global-model-${model.version}.pt"`);
    res.setHeader('Content-Type', 'application/octet-stream');
    fs.createReadStream(modelFilePath).pipe(res);
  } catch (err) {
    next(err);
  }
}

/**
 * POST /api/client/submit
 * Accepts uploaded .pt file, creates submission record, triggers ARC calculation.
 */
async function submitUpdate(req, res, next) {
  try {
    const clientProfile = req.user.clientProfile;
    if (!clientProfile) {
      return res.status(400).json({ error: 'No client profile found' });
    }

    if (!req.file) {
      return res.status(400).json({ error: 'No model file uploaded (.pt required)' });
    }

    const { roundId, localAccuracy, localLoss, samplesTrained, epochs, batchSize, learningRate, trainingDurationSeconds } = req.body;

    // Verify round eligibility via lifecycle service
    const resolved = await resolveEligibleRoundForClient(clientProfile.clientId, roundId);
    if (!resolved.round) {
      return res.status(404).json({ error: 'Training round not found' });
    }
    const round = resolved.round;

    if (round.status !== 'ACTIVE') {
      return res.status(400).json({ error: `Round #${round.roundNumber} is not accepting submissions (status: ${round.status})` });
    }

    if (!resolved.isTargeted) {
      return res.status(403).json({ error: `Client ${clientProfile.clientId} is not designated for Round #${round.roundNumber}.` });
    }

    if (resolved.hasSubmitted) {
      return res.status(409).json({
        error: `Client ${clientProfile.clientId} has already submitted an update for Round #${round.roundNumber}. Duplicate submissions are not allowed.`,
        submissionId: resolved.submission?.id,
      });
    }

    // Compute file checksum for integrity verification
    const fileBuffer = fs.readFileSync(req.file.path);
    const checksum = crypto.createHash('sha256').update(fileBuffer).digest('hex');

    // Relative path for storage (portable)
    const relPath = path.relative(config.storage.root, req.file.path).replace(/\\/g, '/');

    broadcastTrainingProgress(clientProfile.clientId, {
      roundId: round.id,
      roundNumber: round.roundNumber,
      currentStage: 'SUBMITTING_UPDATE',
      progress: 95,
      status: 'SUBMITTING',
    });

    const submission = await prisma.clientSubmission.create({
      data: {
        roundId: round.id,
        clientId: clientProfile.clientId,
        localAccuracy: localAccuracy ? parseFloat(localAccuracy) : null,
        localLoss: localLoss ? parseFloat(localLoss) : null,
        samplesTrained: samplesTrained ? parseInt(samplesTrained) : null,
        epochs: epochs ? parseInt(epochs) : null,
        batchSize: batchSize ? parseInt(batchSize) : null,
        learningRate: learningRate ? parseFloat(learningRate) : null,
        trainingDurationSeconds: trainingDurationSeconds ? parseFloat(trainingDurationSeconds) : null,
        storagePath: relPath,
        checksumSha256: checksum,
      },
    });

    // Trigger ARC calculation asynchronously (don't block the response)
    const telemetry = req.body.telemetry ? JSON.parse(req.body.telemetry) : {};
    try {
      const arcResult = await mlServiceClient.calculateARC({
        submissionId: submission.id,
        clientId: clientProfile.clientId,
        roundId,
        modelPath: req.file.path,
        localAccuracy: parseFloat(localAccuracy || 0),
        localLoss: parseFloat(localLoss || 0),
        samplesTrained: parseInt(samplesTrained || 0),
        telemetry,
      });

      if (arcResult && arcResult.overall_arc_score !== undefined) {
        // Save ARC score to DB
        await prisma.aRCScore.create({
          data: {
            roundId: round.id,
            submissionId: submission.id,
            clientId: clientProfile.clientId,
            relevanceScore: arcResult.relevance_score ?? 0.0,
            resourceScore: arcResult.resource_score ?? 0.0,
            communicationScore: arcResult.communication_score !== null && arcResult.communication_score !== undefined ? arcResult.communication_score : 70.0,
            overallArcScore: arcResult.overall_arc_score ?? 65.0,
            qualityScore: arcResult.quality_score,
            diversityScore: arcResult.diversity_score,
            memoryScore: arcResult.memory_score,
            processingScore: arcResult.processing_score,
            capacityScore: arcResult.capacity_score,
            bandwidthScore: arcResult.bandwidth_score,
            latencyScore: arcResult.latency_score,
            reliabilityScore: arcResult.reliability_score,
            formulaVersion: '1.0',
            weightsUsedJson: arcResult.weights_used,
            rawTelemetryJson: telemetry,
          },
        });
        // Cache overall score on the submission record
        await prisma.clientSubmission.update({
          where: { id: submission.id },
          data: { arcScore: arcResult.overall_arc_score },
        });
      }
    } catch (err) {
      console.error(`[submitUpdate] ARC calculation failed for submission ${submission.id}:`, err.message);
    }

    // Update client's last active time
    await prisma.clientProfile.update({
      where: { id: clientProfile.id },
      data: { lastActiveAt: new Date() },
    });

    // Fetch total submissions count for this round to broadcast accurate updatesReceivedCount
    const [totalUpdatesCount, totalClientsCount] = await Promise.all([
      prisma.clientSubmission.count({
        where: { roundId: round.id, checksumSha256: { not: null } },
      }),
      prisma.clientProfile.count(),
    ]);
    const targetClientsCount = round.targetClients ? round.targetClients.length : totalClientsCount;
    const fileStat = fs.existsSync(req.file.path) ? fs.statSync(req.file.path) : null;
    const updateSize = fileStat ? fileStat.size : null;

    broadcastTrainingProgress(clientProfile.clientId, {
      roundId: round.id,
      roundNumber: round.roundNumber,
      currentStage: 'UPDATE_RECEIVED',
      progress: 100,
      status: 'COMPLETED',
      submissionId: submission.id,
      checksumSha256: submission.checksumSha256,
    });

    broadcastSubmissionRegistered({
      roundId: round.id,
      roundNumber: round.roundNumber,
      clientId: clientProfile.clientId,
      clientName: clientProfile.name || clientProfile.clientId,
      submissionId: submission.id,
      filename: path.basename(req.file.path),
      storagePath: relPath,
      checksumSha256: checksum,
      fileSizeBytes: updateSize,
      fileSizeMb: updateSize ? (updateSize / (1024 * 1024)).toFixed(2) : null,
      status: submission.status,
      submittedAt: submission.submittedAt,
      localAccuracy: submission.localAccuracy,
      localLoss: submission.localLoss,
      updatesReceivedCount: totalUpdatesCount,
      targetClientsCount,
    });

    res.status(201).json({
      message: 'Submission received. ARC calculation in progress.',
      submission: { id: submission.id, status: submission.status, submittedAt: submission.submittedAt },
    });
  } catch (err) {
    next(err);
  }
}

/**
 * GET /api/client/submissions
 * Returns submission history for this client.
 */
async function getSubmissions(req, res, next) {
  try {
    const clientProfile = req.user.clientProfile;
    if (!clientProfile) return res.status(400).json({ error: 'No client profile' });

    const submissions = await prisma.clientSubmission.findMany({
      where: { clientId: clientProfile.clientId },
      orderBy: { submittedAt: 'desc' },
      include: {
        round: { select: { roundNumber: true, status: true } },
        arcScores: true,
      },
    });

    res.json({ submissions });
  } catch (err) {
    next(err);
  }
}

/**
 * GET /api/client/arc-history
 * Returns all ARC scores for this client, most recent first.
 */
async function getArcHistory(req, res, next) {
  try {
    const clientProfile = req.user.clientProfile;
    if (!clientProfile) return res.status(400).json({ error: 'No client profile' });

    const arcScores = await prisma.aRCScore.findMany({
      where: { clientId: clientProfile.clientId },
      orderBy: { calculatedAt: 'desc' },
      include: {
        round: { select: { roundNumber: true } },
        submission: { select: { submittedAt: true, status: true } },
      },
    });

    res.json({ arcScores });
  } catch (err) {
    next(err);
  }
}

/**
 * Helper: identifies the authenticated client's assigned dataset partition.
 * Does not expose another client's dataset.
 */
function getAssignedDatasetInfo(clientId) {
  const datasetsDir = path.join(config.storage.root, 'datasets');
  const numericMatch = (clientId || '').match(/\d+/);
  const num = numericMatch ? parseInt(numericMatch[0], 10) : 1;

  const candidates = [
    `client_${num}.csv`,
    `client${num}.csv`,
    `client${num}_traffic.csv`,
    path.join('P:', `client_${num}.csv`),
    'benchmark_ciciot.csv'
  ];

  let filename = `client_${num}.csv`;
  let fullPath = path.join(datasetsDir, filename);
  for (const cand of candidates) {
    const candPath = path.isAbsolute(cand) ? cand : path.join(datasetsDir, cand);
    if (fs.existsSync(candPath)) {
      filename = path.basename(candPath);
      fullPath = candPath;
      break;
    }
  }

  const stats = fs.existsSync(fullPath) ? fs.statSync(fullPath) : { size: 0 };
  return {
    filename,
    path: fullPath,
    sizeKb: Math.round(stats.size / 1024),
    assignedTo: clientId,
  };
}

/**
 * GET /api/client/datasets
 * Returns ONLY the authenticated client's assigned dataset partition.
 * Never exposes another client's dataset.
 */
async function getDatasets(req, res, next) {
  try {
    const clientProfile = req.user.clientProfile;
    if (!clientProfile) {
      return res.status(400).json({ error: 'No client profile found' });
    }

    const assigned = getAssignedDatasetInfo(clientProfile.clientId);
    res.json({
      assignedDataset: assigned,
      datasets: [assigned],
    });
  } catch (err) {
    next(err);
  }
}

/**
 * POST /api/client/train
 * Executes local training on the client's assigned partition.
 * Validates client-provided training parameters (epochs, batchSize, learningRate).
 */
async function runLocalTraining(req, res, next) {
  try {
    const clientProfile = req.user.clientProfile;
    if (!clientProfile) return res.status(400).json({ error: 'No client profile' });

    // Determine target training round using universal lifecycle service
    const resolved = await resolveEligibleRoundForClient(clientProfile.clientId, req.body.roundId);
    if (!resolved.round) {
      return res.status(404).json({
        error: req.body.roundId ? `Training round with ID ${req.body.roundId} not found.` : 'No active training round found. Client training requires an active round.',
      });
    }
    const round = resolved.round;

    if (round.status !== 'ACTIVE') {
      return res.status(400).json({ error: `Round #${round.roundNumber} is not active (current status: ${round.status}). Training cannot proceed.` });
    }

    if (!resolved.isTargeted) {
      return res.status(403).json({ error: `Client ${clientProfile.clientId} is not designated for Round #${round.roundNumber}.` });
    }

    if (resolved.hasSubmitted) {
      return res.status(409).json({
        error: `Client ${clientProfile.clientId} has already submitted an update for Round #${round.roundNumber}. Duplicate training runs are not allowed.`,
        submissionId: resolved.submission?.id,
      });
    }

    // Load the round's designated base model
    const baseModel = round.baseModel || await prisma.globalModel.findUnique({ where: { id: round.baseModelId } });
    if (!baseModel) {
      return res.status(404).json({ error: `Base model for Round #${round.roundNumber} not found.` });
    }

    const globalModelPath = path.join(config.storage.root, baseModel.storagePath, 'best.pt');
    if (!fs.existsSync(globalModelPath)) {
      return res.status(404).json({ error: `Base model file not found on disk at ${globalModelPath}` });
    }

    // Verify base model checksum on disk
    const baseModelBuffer = fs.readFileSync(globalModelPath);
    const computedBaseSha = crypto.createHash('sha256').update(baseModelBuffer).digest('hex');
    if (baseModel.checksumSha256 && computedBaseSha.toLowerCase() !== baseModel.checksumSha256.toLowerCase()) {
      return res.status(500).json({
        error: `Base model integrity check failed. Expected: ${baseModel.checksumSha256}, Actual: ${computedBaseSha}`,
      });
    }

    // Allow user request to specify training hyperparameters, falling back to round defaults, then system defaults
    const roundHp = (round.hyperparameters && typeof round.hyperparameters === 'object') ? round.hyperparameters : {};
    const epochs = req.body.epochs !== undefined ? req.body.epochs : (roundHp.epochs !== undefined ? roundHp.epochs : 2);
    const batchSize = req.body.batchSize !== undefined ? req.body.batchSize : (roundHp.batchSize !== undefined ? roundHp.batchSize : 256);
    const learningRate = req.body.learningRate !== undefined ? req.body.learningRate : (roundHp.learningRate !== undefined ? roundHp.learningRate : 0.0001);
    const maxSamples = req.body.max_samples !== undefined && req.body.max_samples !== null ? Number(req.body.max_samples) : null;

    // Strict validation of training parameters
    const errors = {};
    const parsedEpochs = Number(epochs);
    const parsedBatchSize = Number(batchSize);
    const parsedLearningRate = Number(learningRate);

    if (!Number.isInteger(parsedEpochs) || parsedEpochs < 1 || parsedEpochs > 50) {
      errors.epochs = 'Local Epochs must be an integer between 1 and 50.';
    }
    if (!Number.isInteger(parsedBatchSize) || parsedBatchSize < 16 || parsedBatchSize > 1024) {
      errors.batchSize = 'Batch Size must be an integer between 16 and 1024.';
    }
    if (isNaN(parsedLearningRate) || parsedLearningRate <= 0 || parsedLearningRate > 1.0) {
      errors.learningRate = 'Learning Rate must be a valid positive float between 0.00001 and 1.0.';
    }

    if (Object.keys(errors).length > 0) {
      return res.status(400).json({
        error: 'Invalid training parameters.',
        validationErrors: errors,
      });
    }

    const assigned = getAssignedDatasetInfo(clientProfile.clientId);
    const datasetPath = assigned.path;

    if (!fs.existsSync(datasetPath)) {
      return res.status(404).json({ error: `Assigned dataset ${assigned.filename} not found on disk.` });
    }

    // Set HTTP socket timeout to 10 minutes to support large datasets / multi-epoch training
    req.setTimeout(600_000);
    res.setTimeout(600_000);

    // Broadcast downloading and verification of global model
    broadcastTrainingProgress(clientProfile.clientId, {
      roundId: round.id,
      roundNumber: round.roundNumber,
      currentStage: 'DOWNLOADING_GLOBAL_MODEL',
      progress: 15,
      status: 'RUNNING',
      baseModelVersion: baseModel.version,
      baseModelChecksum: computedBaseSha,
      message: `Verified global base model checkpoint ${baseModel.version} (${computedBaseSha.substring(0, 16)}...)`,
    });

    // Stream genuine PyTorch edge training loop events
    const trainingResult = await mlServiceClient.streamLocalTraining({
      dataset_path: datasetPath,
      global_model_path: globalModelPath,
      epochs: parsedEpochs,
      batch_size: parsedBatchSize,
      learning_rate: parsedLearningRate,
      max_samples: maxSamples,
      freeze_bn_stats: true,
    }, (streamEvent) => {
      broadcastTrainingProgress(clientProfile.clientId, {
        roundId: round.id,
        roundNumber: round.roundNumber,
        currentStage: streamEvent.stage || 'TRAINING_EPOCH_1',
        progress: streamEvent.progress || 50,
        currentEpoch: streamEvent.epoch || 1,
        totalEpochs: streamEvent.total_epochs || parsedEpochs,
        currentBatch: streamEvent.batch,
        totalBatches: streamEvent.total_batches,
        batchLoss: streamEvent.batch_loss,
        runningLoss: streamEvent.running_loss,
        elapsedSeconds: streamEvent.elapsed_seconds,
        status: 'RUNNING',
        message: streamEvent.message,
        isBatchEvent: streamEvent.event === 'BATCH_PROGRESS' || Boolean(streamEvent.batch),
      });
    });

    // Broadcast local training completion
    broadcastTrainingProgress(clientProfile.clientId, {
      roundId: round.id,
      roundNumber: round.roundNumber,
      currentStage: 'GENERATING_MODEL_UPDATE',
      progress: 88,
      status: 'TRAINED',
      epochHistory: trainingResult.epoch_history,
      localLoss: trainingResult.loss,
      localAccuracy: trainingResult.accuracy,
      updatePath: trainingResult.update_path,
      updateId: trainingResult.update_id,
      trainingDurationSeconds: trainingResult.training_duration_seconds,
      message: `PyTorch local edge fine-tuning completed in ${trainingResult.training_duration_seconds}s. Model update checkpoint generated.`,
    });

    res.json({
      success: true,
      dataset: assigned.filename,
      assignedClientId: clientProfile.clientId,
      baseModelVersion: baseModel.version,
      baseModelId: baseModel.id,
      baseModelChecksum: computedBaseSha,
      roundId: round.id,
      roundNumber: round.roundNumber,
      epochs: parsedEpochs,
      batchSize: parsedBatchSize,
      learningRate: parsedLearningRate,
      ...trainingResult,
    });
  } catch (err) {
    next(err);
  }
}

/**
 * POST /api/client/submit-trained
 * Submits the model update produced by the local training session directly.
 * Preserves genuine client-provided parameters and measured telemetry.
 */
async function submitTrainedUpdate(req, res, next) {
  try {
    const clientProfile = req.user.clientProfile;
    if (!clientProfile) return res.status(400).json({ error: 'No client profile found' });

    const {
      roundId,
      updatePath,
      localAccuracy,
      localLoss,
      samplesTrained,
      epochs,
      batchSize,
      learningRate,
      trainingDurationSeconds,
      resourceMetrics,
      qualityMetrics,
      diversityMetrics,
      commMetrics,
    } = req.body;

    // Validate parameters
    const parsedEpochs = epochs !== undefined ? Number(epochs) : 2;
    const parsedBatchSize = batchSize !== undefined ? Number(batchSize) : 128;
    const parsedLearningRate = learningRate !== undefined ? Number(learningRate) : 0.001;

    if (!Number.isInteger(parsedEpochs) || parsedEpochs < 1 || parsedEpochs > 50) {
      return res.status(400).json({ error: 'Local Epochs must be an integer between 1 and 50.' });
    }
    if (!Number.isInteger(parsedBatchSize) || parsedBatchSize < 16 || parsedBatchSize > 1024) {
      return res.status(400).json({ error: 'Batch Size must be an integer between 16 and 1024.' });
    }
    if (isNaN(parsedLearningRate) || parsedLearningRate <= 0 || parsedLearningRate > 1.0) {
      return res.status(400).json({ error: 'Learning Rate must be between 0.00001 and 1.0.' });
    }

    // Verify round eligibility via universal lifecycle service
    const resolved = await resolveEligibleRoundForClient(clientProfile.clientId, roundId);
    if (!resolved.round) {
      return res.status(404).json({ error: 'Training round not found' });
    }
    const round = resolved.round;

    if (round.status !== 'ACTIVE') {
      return res.status(400).json({ error: `Round #${round.roundNumber} is not accepting submissions (status: ${round.status})` });
    }

    if (!resolved.isTargeted) {
      return res.status(403).json({ error: `Client ${clientProfile.clientId} is not designated for Round #${round.roundNumber}.` });
    }

    if (resolved.hasSubmitted) {
      return res.status(409).json({
        error: `Client ${clientProfile.clientId} has already submitted an update for Round #${round.roundNumber}. Duplicate submissions are not allowed.`,
        submissionId: resolved.submission?.id,
      });
    }

    if (!fs.existsSync(updatePath)) return res.status(404).json({ error: 'Update artifact not found on disk.' });

    const fileBuffer = fs.readFileSync(updatePath);
    const checksum = crypto.createHash('sha256').update(fileBuffer).digest('hex');
    const relPath = path.relative(config.storage.root, updatePath).replace(/\\/g, '/');

    // Save client submission with client-provided parameters
    broadcastTrainingProgress(clientProfile.clientId, {
      roundId: round.id,
      roundNumber: round.roundNumber,
      currentStage: 'SUBMITTING_UPDATE',
      progress: 95,
      status: 'SUBMITTING',
    });

    const submission = await prisma.clientSubmission.create({
      data: {
        roundId: round.id,
        clientId: clientProfile.clientId,
        localAccuracy: localAccuracy ? parseFloat(localAccuracy) : null,
        localLoss: localLoss ? parseFloat(localLoss) : null,
        samplesTrained: samplesTrained ? parseInt(samplesTrained) : null,
        epochs: parsedEpochs,
        batchSize: parsedBatchSize,
        learningRate: parsedLearningRate,
        trainingDurationSeconds: trainingDurationSeconds ? parseFloat(trainingDurationSeconds) : null,
        storagePath: relPath,
        checksumSha256: checksum,
      },
    });

    // Fetch official ARC formula coefficients configured for this round (or global config fallback)
    let arcConfigToUse;
    let formulaVersion = '1.0';
    if (round.arcWeights && typeof round.arcWeights === 'object') {
      const lR = Number(round.arcWeights.lambdaRelevance ?? round.arcWeights.lambda_relevance ?? 0.40);
      const lS = Number(round.arcWeights.lambdaResource ?? round.arcWeights.lambda_resource ?? 0.30);
      const lC = Number(round.arcWeights.lambdaCommunication ?? round.arcWeights.lambda_communication ?? 0.30);

      // Validate: non-negative and sum to 1.0
      if (lR < 0 || lS < 0 || lC < 0) {
        return res.status(400).json({ error: 'Round ARC coefficients must be non-negative.' });
      }
      const sum = lR + lS + lC;
      if (Math.abs(sum - 1.0) > 0.01) {
        return res.status(400).json({ error: `Round ARC coefficients must sum to 1.0 (current sum: ${sum.toFixed(4)})` });
      }

      arcConfigToUse = {
        lambdaRelevance: lR,
        lambdaResource: lS,
        lambdaCommunication: lC,
        ...round.arcWeights,
      };
      formulaVersion = `round-${round.roundNumber}`;
    } else {
      const activeArcConfig = await prisma.aRCConfig.findFirst({
        where: { isActive: true },
        orderBy: { version: 'desc' },
      });
      arcConfigToUse = activeArcConfig || { lambdaRelevance: 0.40, lambdaResource: 0.30, lambdaCommunication: 0.30 };
      formulaVersion = activeArcConfig ? `v${activeArcConfig.version}` : '1.0';
    }

    // Compile genuine measured and client-reported telemetry (distinguishing provenance)
    const parameterSources = req.body.parameterSources || {};
    const telemetry = {
      quality_score: qualityMetrics?.quality_score ?? null,
      diversity_score: diversityMetrics?.diversity_score ?? null,
      cpu_idle_pct: resourceMetrics?.cpu_idle_pct ?? null,
      total_ram_gb: resourceMetrics?.total_ram_gb ?? null,
      available_ram_gb: resourceMetrics?.available_ram_gb ?? null,
      throughput_samples_per_sec: resourceMetrics?.throughput_samples_per_sec ?? null,
      missing_ratio: qualityMetrics?.missing_ratio ?? null,
      inf_ratio: qualityMetrics?.inf_ratio ?? null,
      preprocessing_success: qualityMetrics?.preprocessing_success ?? 1.0,
      class_count: diversityMetrics?.class_count ?? null,
      entropy: diversityMetrics?.entropy ?? null,
      minority_class_coverage: diversityMetrics?.minority_class_coverage ?? null,
      bandwidth_mbps: commMetrics?.bandwidth_mbps ?? null,
      latency_ms: commMetrics?.latency_ms ?? null,
      reliability_pct: commMetrics?.reliability_pct ?? null,
      parameterSources,
      trainingParameters: {
        epochs: parsedEpochs,
        batchSize: parsedBatchSize,
        learningRate: parsedLearningRate,
      },
    };

    try {
      const arcResult = await mlServiceClient.calculateARC({
        submissionId: submission.id,
        clientId: clientProfile.clientId,
        roundId: round.id,
        modelPath: updatePath,
        localAccuracy: parseFloat(localAccuracy || 0),
        localLoss: parseFloat(localLoss || 0),
        samplesTrained: parseInt(samplesTrained || 0),
        arc_config: arcConfigToUse,
        quality_metrics: qualityMetrics,
        diversity_metrics: diversityMetrics,
        resource_metrics: resourceMetrics,
        comm_metrics: commMetrics,
        telemetry,
      });

      if (arcResult && arcResult.overall_arc_score !== undefined) {
        await prisma.aRCScore.create({
          data: {
            roundId: round.id,
            submissionId: submission.id,
            clientId: clientProfile.clientId,
            relevanceScore: arcResult.relevance_score ?? 0.0,
            resourceScore: arcResult.resource_score ?? 0.0,
            communicationScore: arcResult.communication_score !== null && arcResult.communication_score !== undefined ? arcResult.communication_score : 70.0,
            overallArcScore: arcResult.overall_arc_score ?? 65.0,
            qualityScore: arcResult.quality_score,
            diversityScore: arcResult.diversity_score,
            memoryScore: arcResult.memory_score,
            processingScore: arcResult.processing_score,
            capacityScore: arcResult.capacity_score,
            bandwidthScore: arcResult.bandwidth_score,
            latencyScore: arcResult.latency_score,
            reliabilityScore: arcResult.reliability_score,
            formulaVersion: formulaVersion,
            weightsUsedJson: arcResult.weights_used || arcConfigToUse,
            rawTelemetryJson: telemetry,
          },
        });
        await prisma.clientSubmission.update({
          where: { id: submission.id },
          data: { arcScore: arcResult.overall_arc_score },
        });
      }
    } catch (err) {
      console.error('[submitTrainedUpdate] ARC calculation error:', err.message);
    }

    await prisma.clientProfile.update({
      where: { id: clientProfile.id },
      data: { lastActiveAt: new Date() },
    });

    // Fetch total submissions count for this round to broadcast accurate updatesReceivedCount
    const [totalUpdatesCount, totalClientsCount] = await Promise.all([
      prisma.clientSubmission.count({
        where: { roundId: round.id, checksumSha256: { not: null } },
      }),
      prisma.clientProfile.count(),
    ]);
    const targetClientsCount = round.targetClients ? round.targetClients.length : totalClientsCount;
    const fileStat = fs.existsSync(updatePath) ? fs.statSync(updatePath) : null;
    const updateSize = fileStat ? fileStat.size : null;

    broadcastTrainingProgress(clientProfile.clientId, {
      roundId: round.id,
      roundNumber: round.roundNumber,
      currentStage: 'UPDATE_RECEIVED',
      progress: 100,
      status: 'COMPLETED',
      submissionId: submission.id,
      checksumSha256: submission.checksumSha256,
    });

    broadcastSubmissionRegistered({
      roundId: round.id,
      roundNumber: round.roundNumber,
      clientId: clientProfile.clientId,
      clientName: clientProfile.name || clientProfile.clientId,
      submissionId: submission.id,
      filename: path.basename(updatePath),
      storagePath: relPath,
      checksumSha256: checksum,
      fileSizeBytes: updateSize,
      fileSizeMb: updateSize ? (updateSize / (1024 * 1024)).toFixed(2) : null,
      status: submission.status,
      submittedAt: submission.submittedAt,
      localAccuracy: submission.localAccuracy,
      localLoss: submission.localLoss,
      updatesReceivedCount: totalUpdatesCount,
      targetClientsCount,
    });

    res.status(201).json({
      message: 'Submission received and recorded. ARC calculation triggered.',
      submission: {
        id: submission.id,
        status: submission.status,
        submittedAt: submission.submittedAt,
        checksumSha256: submission.checksumSha256,
      },
    });
  } catch (err) {
    next(err);
  }
}

/**
 * GET /api/client/config
 * Retrieves the client's saved configuration (hyperparameters, ARC inputs, autoSubmit preference).
 */
async function getClientConfig(req, res, next) {
  try {
    const clientProfile = req.user.clientProfile;
    if (!clientProfile) {
      return res.status(400).json({ error: 'No client profile found' });
    }

    const currentProfile = await prisma.clientProfile.findUnique({
      where: { id: clientProfile.id },
    });

    const hw = (currentProfile?.hardwareSpecs && typeof currentProfile.hardwareSpecs === 'object')
      ? currentProfile.hardwareSpecs
      : {};

    const savedConfig = hw.savedConfig || {
      epochs: 2,
      batchSize: 256,
      learningRate: 0.0001,
      qualityScore: 98.5,
      diversityScore: 85.0,
      availableRamGb: 8.0,
      totalRamGb: 16.0,
      cpuIdlePct: 65.0,
      throughputSamplesPerSec: 500.0,
      bandwidthMbps: 35.0,
      latencyMs: 40.0,
      reliabilityPct: 99.5,
      autoSubmit: true,
    };

    const paramSources = hw.paramSources || {
      epochs: 'MANUAL',
      batchSize: 'MANUAL',
      learningRate: 'MANUAL',
      qualityScore: 'MEASURED',
      diversityScore: 'MEASURED',
      availableRamGb: 'MEASURED',
      cpuIdlePct: 'MEASURED',
      throughputSamplesPerSec: 'MEASURED',
      bandwidthMbps: 'MEASURED',
      latencyMs: 'MEASURED',
      reliabilityPct: 'MEASURED',
    };

    res.json({
      config: savedConfig,
      paramSources,
      clientId: clientProfile.clientId,
    });
  } catch (err) {
    next(err);
  }
}

/**
 * PUT /api/client/config
 * Persists the client's configuration snapshot in PostgreSQL so Training Studio and Monitor stay synchronized.
 */
async function updateClientConfig(req, res, next) {
  try {
    const clientProfile = req.user.clientProfile;
    if (!clientProfile) {
      return res.status(400).json({ error: 'No client profile found' });
    }

    const { config: newConfig, paramSources } = req.body;
    if (!newConfig || typeof newConfig !== 'object') {
      return res.status(400).json({ error: 'Config object required' });
    }

    const currentProfile = await prisma.clientProfile.findUnique({
      where: { id: clientProfile.id },
    });

    const hw = (currentProfile?.hardwareSpecs && typeof currentProfile.hardwareSpecs === 'object')
      ? currentProfile.hardwareSpecs
      : {};

    const mergedHw = {
      ...hw,
      savedConfig: {
        ...(hw.savedConfig || {}),
        ...newConfig,
      },
      paramSources: paramSources || hw.paramSources || {},
      lastConfigUpdate: new Date().toISOString(),
    };

    await prisma.clientProfile.update({
      where: { id: clientProfile.id },
      data: {
        hardwareSpecs: mergedHw,
      },
    });

    res.json({
      message: 'Configuration saved successfully.',
      config: mergedHw.savedConfig,
      paramSources: mergedHw.paramSources,
    });
  } catch (err) {
    next(err);
  }
}

module.exports = {
  getDashboard,
  getGlobalModel,
  downloadGlobalModel,
  submitUpdate,
  getSubmissions,
  getArcHistory,
  getDatasets,
  runLocalTraining,
  submitTrainedUpdate,
  getClientConfig,
  updateClientConfig,
};
