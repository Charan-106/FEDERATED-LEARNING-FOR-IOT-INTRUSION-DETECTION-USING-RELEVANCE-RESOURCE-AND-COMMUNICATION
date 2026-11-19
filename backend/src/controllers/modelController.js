// Model Controller - Global model lifecycle management
// Handles versioning, publishing, candidate conversion, download, and rollback

const path = require('path');
const fs = require('fs');
const crypto = require('crypto');
const prisma = require('../config/database');
const config = require('../config');
const modelLifecycleService = require('../services/modelLifecycleService');

/**
 * GET /api/models/references
 * Return the authoritative three-model references:
 * 1. INITIAL_BASELINE (initial-v1)
 * 2. PREVIOUS_BEST (highest validated historical performer)
 * 3. ACTIVE_GLOBAL (current production model for FL rounds)
 */
async function getModelReferences(req, res, next) {
  try {
    const references = await modelLifecycleService.getAuthoritativeReferences();
    res.json(references);
  } catch (err) {
    next(err);
  }
}

/**
 * GET /api/models/compare
 * Compare two models side-by-side with metric deltas and benchmark equivalence checks.
 */
async function compareModelsController(req, res, next) {
  try {
    const { modelA, modelB } = req.query;
    if (!modelA || !modelB) {
      return res.status(400).json({ error: 'Both modelA and modelB query parameters are required' });
    }
    const comparison = await modelLifecycleService.compareTwoModels(modelA, modelB);
    res.json(comparison);
  } catch (err) {
    next(err);
  }
}

/**
 * GET /api/models/details/:modelId
 * Return full metadata, evaluation history, and lineage for a specific model version.
 */
async function getModelDetails(req, res, next) {
  try {
    const { modelId } = req.params;
    const model = await prisma.globalModel.findUnique({
      where: { id: modelId },
      include: {
        evaluations: { orderBy: { evaluatedAt: 'desc' } },
        baseRounds: { orderBy: { roundNumber: 'desc' }, take: 10 },
        resultingRounds: true,
        parentModel: true,
        childModels: true,
      },
    });
    if (!model) {
      return res.status(404).json({ error: 'Model version not found' });
    }
    res.json({ model });
  } catch (err) {
    next(err);
  }
}

/**
 * GET /api/models/current
 * Return active global model metadata and recent evaluation reports.
 */
async function getCurrentModel(req, res, next) {
  try {
    const model = await prisma.globalModel.findFirst({
      where: { isActive: true },
      include: {
        evaluations: { orderBy: { evaluatedAt: 'desc' }, take: 3 },
      },
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
 * GET /api/models/versions
 * List all global model versions with base round counts, evaluations, and computed lifecycle roles.
 */
async function getVersions(req, res, next) {
  try {
    const models = await modelLifecycleService.getAllModelVersions();
    const references = await modelLifecycleService.getAuthoritativeReferences();
    res.json({ models, references });
  } catch (err) {
    next(err);
  }
}

/**
 * GET /api/models/:modelId/download
 * Stream a specific model version's best.pt file directly from disk.
 */
async function downloadVersion(req, res, next) {
  try {
    const { modelId } = req.params;
    const model = await prisma.globalModel.findUnique({ where: { id: modelId } });

    if (!model) return res.status(404).json({ error: 'Model not found' });

    const modelFilePath = path.join(config.storage.root, model.storagePath, 'best.pt');
    if (!fs.existsSync(modelFilePath)) {
      return res.status(404).json({ error: `Model file not found on disk: ${modelFilePath}` });
    }

    res.setHeader('Content-Disposition', `attachment; filename="model-${model.version}.pt"`);
    res.setHeader('Content-Type', 'application/octet-stream');
    fs.createReadStream(modelFilePath).pipe(res);
  } catch (err) {
    next(err);
  }
}

/**
 * POST /api/models/publish-candidate
 * Publish a merged candidate model from an AggregationJob as a new GlobalModel version.
 */
/**
 * POST /api/models/publish-candidate
 * Atomically promotes a candidate model from an AggregationJob to a production GlobalModel.
 * Applies regression safety gate, copies checkpoints & assets, links parent & source round,
 * updates SystemModelReference, and updates PREVIOUS_BEST if criteria are met.
 */
async function publishCandidate(req, res, next) {
  try {
    const {
      roundId,
      jobId,
      version: customVersion,
      name: customName,
      allowRegression,
      justification,
    } = req.body;

    const result = await modelLifecycleService.publishCandidateModel({
      roundId,
      jobId,
      targetVersion: customVersion,
      targetName: customName,
      allowRegression: allowRegression === true,
      justification,
      adminUser: req.user,
    });

    res.json({
      message: `Candidate model published successfully as active global model ${result.model.version}!`,
      model: result.model,
      authoritativeReferences: result.references,
      isNewPreviousBest: result.isNewBest,
    });
  } catch (err) {
    if (err.status === 422) {
      return res.status(422).json({
        error: err.message,
        ...err.details,
      });
    }
    next(err);
  }
}

/**
 * POST /api/models/publish-candidate-and-start-round
 * Atomically publishes candidate model AND immediately initializes & starts the next federated round.
 * Closes the complete round lifecycle loop seamlessly without manual intervention.
 */
async function publishCandidateAndStartRound(req, res, next) {
  try {
    const {
      roundId,
      jobId,
      version: customVersion,
      name: customName,
      allowRegression,
      justification,
      nextRoundName,
      clientIds,
    } = req.body;

    // 1. Publish candidate model
    const result = await modelLifecycleService.publishCandidateModel({
      roundId,
      jobId,
      targetVersion: customVersion,
      targetName: customName,
      allowRegression: allowRegression === true,
      justification,
      adminUser: req.user,
    });

    // 2. Identify target clients for next round
    let targetClients = clientIds;
    if (!targetClients || !Array.isArray(targetClients) || targetClients.length === 0) {
      const activeProfiles = await prisma.clientProfile.findMany({
        where: { status: 'ACTIVE' },
        select: { clientId: true },
      });
      targetClients = activeProfiles.map((c) => c.clientId);
    }

    // 3. Increment round number
    const lastRound = await prisma.trainingRound.findFirst({
      orderBy: { roundNumber: 'desc' },
    });
    const nextRoundNumber = lastRound ? lastRound.roundNumber + 1 : 1;

    // 4. Create and activate next round
    const nextRound = await prisma.trainingRound.create({
      data: {
        roundNumber: nextRoundNumber,
        name: nextRoundName || `Federated Round #${nextRoundNumber}`,
        description: `Automated lifecycle progression based on active global model ${result.model.version}`,
        baseModelId: result.model.id,
        baseModelChecksum: result.model.checksumSha256,
        status: 'ACTIVE',
        clientCount: targetClients.length,
        targetClients,
        hyperparameters: {
          epochs: 2,
          batchSize: 256,
          learningRate: 0.0001,
          timeoutSeconds: 300,
        },
        arcWeights: {
          lambdaRelevance: 0.40,
          lambdaResource: 0.30,
          lambdaCommunication: 0.30,
        },
        aggregationStrategy: 'ARC_WEIGHTED',
        startedAt: new Date(),
      },
      include: {
        baseModel: { select: { version: true, name: true, baselineAccuracy: true, checksumSha256: true } },
      },
    });

    // 5. Audit Log
    await prisma.auditLog.create({
      data: {
        userId: req.user.id,
        action: 'PUBLISH_AND_START_ROUND',
        resourceType: 'TrainingRound',
        resourceId: nextRound.id,
        detailsJson: {
          publishedModelVersion: result.model.version,
          nextRoundNumber,
          nextRoundId: nextRound.id,
          targetClients,
        },
        ipAddress: req.ip,
      },
    });

    res.json({
      message: `Model ${result.model.version} published and Federated Round #${nextRoundNumber} started successfully!`,
      model: result.model,
      authoritativeReferences: result.references,
      isNewPreviousBest: result.isNewBest,
      nextRound,
    });
  } catch (err) {
    if (err.status === 422) {
      return res.status(422).json({
        error: err.message,
        ...err.details,
      });
    }
    next(err);
  }
}

/**
 * POST /api/models/:modelId/publish
 * Set an existing model as the active global model (atomic reference sync).
 */
async function publishModel(req, res, next) {
  try {
    const { modelId } = req.params;
    const reason = req.body?.reason || `Activated existing model version by administrator`;

    const result = await modelLifecycleService.rollbackToModel({
      modelId,
      reason,
      adminUser: req.user,
    });

    res.json({
      message: `Model ${result.activeModel.version} is now the active global model`,
      model: result.activeModel,
      authoritativeReferences: result.references,
    });
  } catch (err) {
    next(err);
  }
}

/**
 * POST /api/models/:modelId/rollback
 * Activate a previous model version (instant zero-downtime rollback with authoritative tracking).
 */
async function rollbackModel(req, res, next) {
  try {
    const { modelId } = req.params;
    const reason = req.body?.reason || 'Reverted by administrator via Model Lifecycle Management';

    const result = await modelLifecycleService.rollbackToModel({
      modelId,
      reason,
      adminUser: req.user,
    });

    res.json({
      message: `Successfully rolled back active global model to ${result.model.version}`,
      model: result.model,
      previousActive: result.previousActive,
      authoritativeReferences: result.references,
      reason: result.reason,
    });
  } catch (err) {
    next(err);
  }
}

/**
 * POST /api/models/reject-candidate
 * Explicitly rejects a candidate model from an aggregation job.
 * Leaves the active global model unchanged.
 * Records rejection reason and audit log.
 */
async function rejectCandidate(req, res, next) {
  try {
    const { roundId, jobId, reason } = req.body;
    const where = {};
    if (jobId) where.id = jobId;
    if (roundId) where.roundId = roundId;
    where.status = 'COMPLETED';

    const job = await prisma.aggregationJob.findFirst({
      where,
      orderBy: { completedAt: 'desc' },
      include: { round: true },
    });

    if (!job) {
      return res.status(404).json({ error: 'No candidate model job found to reject' });
    }

    // Active model remains active
    const activeModel = await prisma.globalModel.findFirst({ where: { isActive: true } });

    await prisma.auditLog.create({
      data: {
        userId: req.user.id,
        action: 'REJECT_CANDIDATE_MODEL',
        resourceType: 'AggregationJob',
        resourceId: job.id,
        detailsJson: {
          roundId: job.roundId,
          jobId: job.id,
          candidatePath: job.candidateModelPath,
          candidateChecksum: job.candidateChecksum,
          evaluation: job.evaluationReportJson,
          rejectionReason: reason || 'Candidate model performance fell below required operational threshold',
          activeModelRetained: activeModel ? activeModel.version : 'none',
        },
        ipAddress: req.ip,
      },
    });

    res.json({
      success: true,
      message: 'Candidate model rejected. The active global model remains unchanged.',
      activeModel: activeModel ? { id: activeModel.id, version: activeModel.version } : null,
      rejectionReason: reason || 'Candidate model performance fell below required threshold',
    });
  } catch (err) {
    next(err);
  }
}

module.exports = {
  getModelReferences,
  compareModelsController,
  getModelDetails,
  getCurrentModel,
  getVersions,
  downloadVersion,
  publishCandidate,
  publishCandidateAndStartRound,
  rejectCandidate,
  publishModel,
  rollbackModel,
};
