// Admin Controller - Platform management and oversight endpoints
// Handles all ADMIN role operations: clients, rounds, submissions, aggregation

const prisma = require('../config/database');
const config = require('../config');
const mlServiceClient = require('../services/mlServiceClient');
const { auditClientDatasets } = require('../services/datasetAuditService');

/**
 * GET /api/admin/dashboard
 * Aggregate platform stats for the admin overview.
 */
async function getDashboard(req, res, next) {
  try {
    const [
      totalClients,
      activeClients,
      totalSubmissions,
      pendingSubmissions,
      completedRounds,
      activeModel,
    ] = await Promise.all([
      prisma.clientProfile.count(),
      prisma.clientProfile.count({ where: { status: 'ACTIVE' } }),
      prisma.clientSubmission.count(),
      prisma.clientSubmission.count({ where: { status: 'PENDING' } }),
      prisma.trainingRound.count({ where: { status: 'COMPLETED' } }),
      prisma.globalModel.findFirst({ where: { isActive: true } }),
    ]);

    const latestRound = await prisma.trainingRound.findFirst({
      orderBy: { roundNumber: 'desc' },
      include: { _count: { select: { submissions: true } } },
    });

    res.json({
      stats: {
        totalClients,
        activeClients,
        totalSubmissions,
        pendingSubmissions,
        completedRounds,
      },
      latestRound,
      activeModel: activeModel
        ? {
            id: activeModel.id,
            version: activeModel.version,
            accuracy: activeModel.baselineAccuracy,
            checksumSha256: activeModel.checksumSha256,
          }
        : null,
    });
  } catch (err) {
    next(err);
  }
}

/**
 * GET /api/admin/clients
 * List all client profiles with latest ARC scores.
 */
async function getClients(req, res, next) {
  try {
    const clients = await prisma.clientProfile.findMany({
      include: {
        user: { select: { id: true, email: true, name: true, createdAt: true } },
        arcScores: {
          orderBy: { calculatedAt: 'desc' },
          take: 1,
        },
        _count: { select: { submissions: true } },
      },
      orderBy: { createdAt: 'asc' },
    });

    res.json({ clients });
  } catch (err) {
    next(err);
  }
}

/**
 * PATCH /api/admin/clients/:clientId/status
 * Toggle ACTIVE/INACTIVE for a client.
 */
async function updateClientStatus(req, res, next) {
  try {
    const { clientId } = req.params;
    const { status } = req.body;

    if (!['ACTIVE', 'INACTIVE'].includes(status)) {
      return res.status(400).json({ error: 'Status must be ACTIVE or INACTIVE' });
    }

    const client = await prisma.clientProfile.update({
      where: { clientId },
      data: { status },
    });

    await prisma.auditLog.create({
      data: {
        userId: req.user.id,
        action: 'UPDATE_CLIENT_STATUS',
        resourceType: 'ClientProfile',
        resourceId: clientId,
        detailsJson: { newStatus: status },
        ipAddress: req.ip,
      },
    });

    res.json({ client });
  } catch (err) {
    next(err);
  }
}

/**
 * GET /api/admin/rounds
 * List all training rounds.
 */
async function getRounds(req, res, next) {
  try {
    const rounds = await prisma.trainingRound.findMany({
      orderBy: { roundNumber: 'desc' },
      include: {
        baseModel: { select: { version: true, baselineAccuracy: true } },
        resultingModel: { select: { version: true, baselineAccuracy: true } },
        _count: { select: { submissions: true } },
      },
    });

    res.json({ rounds });
  } catch (err) {
    next(err);
  }
}

/**
 * GET /api/admin/rounds/:roundId
 * Fetch complete round details including baseModel, resultingModel, targetClients, submissions, aggregationJobs.
 */
async function getRoundDetails(req, res, next) {
  try {
    const { roundId } = req.params;
    const round = await prisma.trainingRound.findUnique({
      where: { id: roundId },
      include: {
        baseModel: true,
        resultingModel: true,
        submissions: {
          include: {
            client: { select: { clientId: true, name: true, hardwareSpecs: true, status: true } },
            arcScores: { orderBy: { calculatedAt: 'desc' }, take: 1 },
          },
          orderBy: { submittedAt: 'desc' },
        },
        aggregationJobs: {
          include: { weights: true },
          orderBy: { createdAt: 'desc' },
          take: 1,
        },
        arcScores: {
          orderBy: { calculatedAt: 'desc' },
        },
      },
    });

    if (!round) {
      return res.status(404).json({ error: 'Training round not found' });
    }

    res.json({ round });
  } catch (err) {
    next(err);
  }
}

/**
 * POST /api/admin/rounds
 * Create a new configurable training round.
 * Accepts: name, description, baseModelId, clientIds, hyperparameters, arcWeights, aggregationStrategy
 * Defaults to status: 'CREATED' (pending, does not auto-start).
 */
async function createRound(req, res, next) {
  try {
    const {
      name,
      description,
      baseModelId,
      hyperparameters,
      arcWeights,
      aggregationStrategy = 'ARC_WEIGHTED',
      status,
      autoStart,
    } = req.body;
    const clientIds = req.body.clientIds || req.body.targetClients || [];

    // 1. Determine base model
    let baseModel;
    if (baseModelId) {
      baseModel = await prisma.globalModel.findUnique({ where: { id: baseModelId } });
      if (!baseModel) {
        return res.status(400).json({ error: `Selected base model ID ${baseModelId} not found` });
      }
    } else {
      const activeRef = await prisma.systemModelReference.findUnique({ where: { id: 'active_references' } });
      if (activeRef?.activeGlobalModelId) {
        baseModel = await prisma.globalModel.findUnique({ where: { id: activeRef.activeGlobalModelId } });
      }
      if (!baseModel) {
        baseModel = await prisma.globalModel.findFirst({ where: { isActive: true } });
      }
      if (!baseModel) {
        return res.status(400).json({ error: 'No active global model found to base round on' });
      }
    }

    // 2. Validate clientIds
    if (clientIds.length > 0) {
      const existingClients = await prisma.clientProfile.findMany({
        where: { clientId: { in: clientIds } },
        select: { clientId: true },
      });
      const foundIds = existingClients.map((c) => c.clientId);
      const invalidIds = clientIds.filter((id) => !foundIds.includes(id));
      if (invalidIds.length > 0) {
        return res.status(400).json({ error: `Invalid client IDs: ${invalidIds.join(', ')}` });
      }
    }

    // 3. Validate ARC weights if provided
    let finalArcWeights = { lambdaRelevance: 0.40, lambdaResource: 0.30, lambdaCommunication: 0.30 };
    if (arcWeights) {
      const { lambdaRelevance, lambdaResource, lambdaCommunication } = arcWeights;
      if (
        typeof lambdaRelevance !== 'number' ||
        typeof lambdaResource !== 'number' ||
        typeof lambdaCommunication !== 'number' ||
        lambdaRelevance < 0 || lambdaResource < 0 || lambdaCommunication < 0 ||
        lambdaRelevance > 1 || lambdaResource > 1 || lambdaCommunication > 1
      ) {
        return res.status(400).json({ error: 'ARC weights must be numbers between 0 and 1' });
      }
      const sum = lambdaRelevance + lambdaResource + lambdaCommunication;
      if (Math.abs(sum - 1.0) > 0.001) {
        return res.status(400).json({ error: `ARC weights must sum to 1.0 (got ${sum.toFixed(4)})` });
      }
      finalArcWeights = { lambdaRelevance, lambdaResource, lambdaCommunication };
    }

    // 4. Validate hyperparameters if provided (Defaults aligned with research protocol: lr=0.0001, epochs=2, batchSize=32)
    let finalHyperparams = { epochs: 2, batchSize: 32, learningRate: 0.0001, timeoutSeconds: 300 };
    if (hyperparameters) {
      const epochs = Number(hyperparameters.epochs);
      const batchSize = Number(hyperparameters.batchSize);
      const learningRate = Number(hyperparameters.learningRate);

      if (!Number.isInteger(epochs) || epochs < 1 || epochs > 50) {
        return res.status(400).json({ error: 'Local epochs must be an integer between 1 and 50' });
      }
      if (!Number.isInteger(batchSize) || batchSize < 1 || batchSize > 1024) {
        return res.status(400).json({ error: 'Batch size must be an integer between 1 and 1024' });
      }
      if (isNaN(learningRate) || learningRate <= 0 || learningRate > 1.0) {
        return res.status(400).json({ error: 'Learning rate must be a positive float <= 1.0' });
      }
      finalHyperparams = {
        epochs,
        batchSize,
        learningRate,
        timeoutSeconds: Number(hyperparameters.timeoutSeconds) || 300,
      };
    }

    // 5. Pre-training Client Dataset Class Distribution Audit
    const datasetAudit = auditClientDatasets(clientIds);

    // 6. Validate aggregation strategy
    const validStrategies = ['ARC_WEIGHTED', 'FEDAVG', 'EQUAL'];
    if (!validStrategies.includes(aggregationStrategy)) {
      return res.status(400).json({ error: `Invalid aggregation strategy. Must be one of: ${validStrategies.join(', ')}` });
    }

    // 7. Auto-increment round number
    const lastRound = await prisma.trainingRound.findFirst({
      orderBy: { roundNumber: 'desc' },
    });
    const roundNumber = lastRound ? lastRound.roundNumber + 1 : 1;

    const initialStatus = (autoStart || status === 'ACTIVE') ? 'ACTIVE' : 'CREATED';

    if (initialStatus === 'ACTIVE') {
      // Conclude prior active rounds to guarantee strictly one active round in the platform
      const previousActive = await prisma.trainingRound.findMany({ where: { status: 'ACTIVE' } });
      for (const prev of previousActive) {
        const subCount = await prisma.clientSubmission.count({ where: { roundId: prev.id } });
        const nextStatus = subCount > 0 ? 'COMPLETED' : 'CANCELLED';
        await prisma.trainingRound.update({
          where: { id: prev.id },
          data: { status: nextStatus, completedAt: new Date() },
        });
      }
    }

    const round = await prisma.trainingRound.create({
      data: {
        roundNumber,
        name: name || `Federated Round #${roundNumber}`,
        description: description || null,
        baseModelId: baseModel.id,
        baseModelChecksum: baseModel.checksumSha256,
        status: initialStatus,
        clientCount: clientIds.length,
        targetClients: clientIds,
        hyperparameters: finalHyperparams,
        arcWeights: finalArcWeights,
        aggregationStrategy,
        startedAt: initialStatus === 'ACTIVE' ? new Date() : null,
      },
      include: {
        baseModel: { select: { version: true, name: true, baselineAccuracy: true, checksumSha256: true } },
      },
    });

    await prisma.auditLog.create({
      data: {
        userId: req.user.id,
        action: 'CREATE_ROUND',
        resourceType: 'TrainingRound',
        resourceId: round.id,
        detailsJson: {
          roundNumber,
          name: round.name,
          baseModelVersion: baseModel.version,
          targetClients: clientIds,
          hyperparameters: finalHyperparams,
          arcWeights: finalArcWeights,
          aggregationStrategy,
          datasetAuditWarnings: datasetAudit.warnings,
          missingAcrossAllClients: datasetAudit.missingAcrossAllClients,
        },
        ipAddress: req.ip,
      },
    });

    res.status(201).json({ round, datasetAudit });
  } catch (err) {
    next(err);
  }
}

/**
 * PUT /api/admin/rounds/:roundId
 * Update an existing round configuration. Allowed ONLY for CREATED (pending) rounds.
 */
async function updateRound(req, res, next) {
  try {
    const { roundId } = req.params;
    const {
      name,
      description,
      clientIds,
      hyperparameters,
      arcWeights,
      aggregationStrategy,
      baseModelId,
    } = req.body;

    const round = await prisma.trainingRound.findUnique({ where: { id: roundId } });
    if (!round) return res.status(404).json({ error: 'Round not found' });

    if (round.status !== 'CREATED') {
      return res.status(400).json({
        error: `Cannot edit round with status '${round.status}'. Only CREATED (pending) rounds can be edited.`,
      });
    }

    const updateData = {};
    if (name !== undefined) updateData.name = name;
    if (description !== undefined) updateData.description = description;

    if (baseModelId) {
      const baseModel = await prisma.globalModel.findUnique({ where: { id: baseModelId } });
      if (!baseModel) return res.status(400).json({ error: 'Selected base model not found' });
      updateData.baseModelId = baseModelId;
    }

    if (Array.isArray(clientIds)) {
      updateData.targetClients = clientIds;
      updateData.clientCount = clientIds.length;
    }

    if (hyperparameters) {
      const epochs = Number(hyperparameters.epochs);
      const batchSize = Number(hyperparameters.batchSize);
      const learningRate = Number(hyperparameters.learningRate);
      if (!Number.isInteger(epochs) || epochs < 1 || epochs > 50) {
        return res.status(400).json({ error: 'Local epochs must be an integer between 1 and 50' });
      }
      if (!Number.isInteger(batchSize) || batchSize < 1 || batchSize > 1024) {
        return res.status(400).json({ error: 'Batch size must be an integer between 1 and 1024' });
      }
      if (isNaN(learningRate) || learningRate <= 0 || learningRate > 1.0) {
        return res.status(400).json({ error: 'Learning rate must be a positive float <= 1.0' });
      }
      updateData.hyperparameters = {
        epochs,
        batchSize,
        learningRate,
        timeoutSeconds: Number(hyperparameters.timeoutSeconds) || 300,
      };
    }

    if (arcWeights) {
      const { lambdaRelevance, lambdaResource, lambdaCommunication } = arcWeights;
      if (
        typeof lambdaRelevance !== 'number' ||
        typeof lambdaResource !== 'number' ||
        typeof lambdaCommunication !== 'number' ||
        lambdaRelevance < 0 || lambdaResource < 0 || lambdaCommunication < 0 ||
        lambdaRelevance > 1 || lambdaResource > 1 || lambdaCommunication > 1
      ) {
        return res.status(400).json({ error: 'ARC weights must be numbers between 0 and 1' });
      }
      const sum = lambdaRelevance + lambdaResource + lambdaCommunication;
      if (Math.abs(sum - 1.0) > 0.001) {
        return res.status(400).json({ error: `ARC weights must sum to 1.0 (got ${sum.toFixed(4)})` });
      }
      updateData.arcWeights = { lambdaRelevance, lambdaResource, lambdaCommunication };
    }

    if (aggregationStrategy) {
      const validStrategies = ['ARC_WEIGHTED', 'FEDAVG', 'EQUAL'];
      if (!validStrategies.includes(aggregationStrategy)) {
        return res.status(400).json({ error: 'Invalid aggregation strategy' });
      }
      updateData.aggregationStrategy = aggregationStrategy;
    }

    const updatedRound = await prisma.trainingRound.update({
      where: { id: roundId },
      data: updateData,
      include: { baseModel: true },
    });

    await prisma.auditLog.create({
      data: {
        userId: req.user.id,
        action: 'UPDATE_ROUND',
        resourceType: 'TrainingRound',
        resourceId: roundId,
        detailsJson: updateData,
        ipAddress: req.ip,
      },
    });

    res.json({ round: updatedRound });
  } catch (err) {
    next(err);
  }
}

/**
 * POST /api/admin/rounds/:roundId/start
 * Start a CREATED (pending) round, transitioning it to ACTIVE.
 */
async function startRound(req, res, next) {
  try {
    const { roundId } = req.params;
    const round = await prisma.trainingRound.findUnique({ where: { id: roundId } });
    if (!round) return res.status(404).json({ error: 'Round not found' });

    if (round.status !== 'CREATED') {
      return res.status(400).json({
        error: `Cannot start round in status '${round.status}'. Only CREATED rounds can be started.`,
      });
    }

    // Conclude any prior active rounds
    const previousActive = await prisma.trainingRound.findMany({ where: { status: 'ACTIVE', id: { not: roundId } } });
    for (const prev of previousActive) {
      const subCount = await prisma.clientSubmission.count({ where: { roundId: prev.id } });
      const nextStatus = subCount > 0 ? 'COMPLETED' : 'CANCELLED';
      await prisma.trainingRound.update({
        where: { id: prev.id },
        data: { status: nextStatus, completedAt: new Date() },
      });
    }

    const updated = await prisma.trainingRound.update({
      where: { id: roundId },
      data: {
        status: 'ACTIVE',
        startedAt: new Date(),
      },
    });

    await prisma.auditLog.create({
      data: {
        userId: req.user.id,
        action: 'START_ROUND',
        resourceType: 'TrainingRound',
        resourceId: roundId,
        detailsJson: { roundNumber: round.roundNumber },
        ipAddress: req.ip,
      },
    });

    res.json({ round: updated, message: `Round #${round.roundNumber} is now ACTIVE.` });
  } catch (err) {
    next(err);
  }
}

/**
 * POST /api/admin/rounds/:roundId/cancel
 * Cancel a pending or active round.
 */
async function cancelRound(req, res, next) {
  try {
    const { roundId } = req.params;
    const { reason } = req.body;

    const round = await prisma.trainingRound.findUnique({ where: { id: roundId } });
    if (!round) return res.status(404).json({ error: 'Round not found' });

    if (['COMPLETED', 'CANCELLED'].includes(round.status)) {
      return res.status(400).json({
        error: `Cannot cancel round in '${round.status}' status.`,
      });
    }

    const updated = await prisma.trainingRound.update({
      where: { id: roundId },
      data: {
        status: 'CANCELLED',
        completedAt: new Date(),
      },
    });

    await prisma.auditLog.create({
      data: {
        userId: req.user.id,
        action: 'CANCEL_ROUND',
        resourceType: 'TrainingRound',
        resourceId: roundId,
        detailsJson: { previousStatus: round.status, reason: reason || 'Cancelled by Admin' },
        ipAddress: req.ip,
      },
    });

    res.json({ round: updated, message: `Round #${round.roundNumber} cancelled successfully.` });
  } catch (err) {
    next(err);
  }
}

/**
 * GET /api/admin/rounds/:roundId/dependencies
 * Pre-deletion/archival dependency analysis.
 * Inspects submissions, candidates, active status, and published models.
 */
async function checkRoundDependencies(req, res, next) {
  try {
    const { roundId } = req.params;
    const round = await prisma.trainingRound.findUnique({
      where: { id: roundId },
      include: {
        baseModel: { select: { id: true, version: true, isActive: true } },
        resultingModel: { select: { id: true, version: true, isActive: true, role: true } },
        submissions: {
          select: { id: true, clientId: true, status: true, submittedAt: true },
        },
        aggregationJobs: {
          select: { id: true, status: true, candidateModelPath: true, candidateChecksum: true, createdAt: true },
        },
      },
    });

    if (!round) {
      return res.status(404).json({ error: 'Training round not found' });
    }

    const isActive = ['ACTIVE', 'AGGREGATING'].includes(round.status);
    const hasResultingModel = Boolean(round.resultingModelId);
    const isResultingModelActive = round.resultingModel?.isActive === true;
    const isCompleted = round.status === 'COMPLETED';
    const isArchived = round.status === 'ARCHIVED';
    const submissionCount = round.submissions.length;
    const aggregationCount = round.aggregationJobs.length;

    // Can delete: Only draft/unstarted (CREATED), CANCELLED, or FAILED rounds with NO resulting model and not currently active
    const canDelete = !isActive && !hasResultingModel && ['CREATED', 'CANCELLED', 'FAILED'].includes(round.status);

    // Can archive: Completed, cancelled, or failed rounds that are not already archived and not currently active
    const canArchive = !isActive && !isArchived && ['COMPLETED', 'CANCELLED', 'FAILED'].includes(round.status);

    let recommendation = 'DELETE';
    let blockReason = null;

    if (isActive) {
      recommendation = 'BLOCK';
      blockReason = `Round is currently in status '${round.status}'. Active training rounds must be completed or cancelled before managing.`;
    } else if (isResultingModelActive) {
      recommendation = 'ARCHIVE_ONLY';
      blockReason = `Round produced global model ${round.resultingModel.version}, which is the currently active production model. This round cannot be deleted.`;
    } else if (isCompleted) {
      recommendation = 'ARCHIVE';
    } else if (canDelete) {
      recommendation = 'DELETE';
    }

    res.json({
      round: {
        id: round.id,
        roundNumber: round.roundNumber,
        name: round.name,
        status: round.status,
        baseModel: round.baseModel,
        resultingModel: round.resultingModel,
        startedAt: round.startedAt,
        completedAt: round.completedAt,
        archivedAt: round.archivedAt,
      },
      dependencies: {
        submissionCount,
        submissions: round.submissions,
        aggregationCount,
        aggregationJobs: round.aggregationJobs,
        hasResultingModel,
        resultingModelVersion: round.resultingModel?.version || null,
        isResultingModelActive,
      },
      permissions: {
        canDelete,
        canArchive,
        recommendation,
        blockReason,
      },
    });
  } catch (err) {
    next(err);
  }
}

/**
 * POST /api/admin/rounds/:roundId/archive
 * Safely archive a completed or cancelled round to historical research records.
 */
async function archiveRound(req, res, next) {
  try {
    const { roundId } = req.params;
    const { reason } = req.body;

    const round = await prisma.trainingRound.findUnique({
      where: { id: roundId },
      include: { resultingModel: true },
    });

    if (!round) return res.status(404).json({ error: 'Round not found' });

    if (['ACTIVE', 'AGGREGATING'].includes(round.status)) {
      return res.status(400).json({
        error: `Cannot archive round with status '${round.status}'. Complete or cancel the round first.`,
      });
    }

    if (round.status === 'ARCHIVED') {
      return res.status(400).json({ error: 'Round is already archived.' });
    }

    const updated = await prisma.trainingRound.update({
      where: { id: roundId },
      data: {
        status: 'ARCHIVED',
        archivedAt: new Date(),
        archivedBy: req.user.id,
        archiveReason: reason || 'Archived by administrator to preserve historical benchmark data',
      },
    });

    await prisma.auditLog.create({
      data: {
        userId: req.user.id,
        action: 'ARCHIVE_ROUND',
        resourceType: 'TrainingRound',
        resourceId: roundId,
        detailsJson: {
          roundNumber: round.roundNumber,
          previousStatus: round.status,
          reason: reason || 'Archived to historical records',
        },
        ipAddress: req.ip,
      },
    });

    res.json({
      message: `Round #${round.roundNumber} has been safely archived to historical research records.`,
      round: updated,
    });
  } catch (err) {
    next(err);
  }
}

/**
 * DELETE /api/admin/rounds/:roundId
 * Delete a round safely.
 * Allowed only for: CREATED (pending), CANCELLED, or FAILED.
 * Disallowed for: ACTIVE (must be cancelled first), COMPLETED (preserves research history), or published rounds.
 */
async function deleteRound(req, res, next) {
  try {
    const { roundId } = req.params;
    const round = await prisma.trainingRound.findUnique({
      where: { id: roundId },
      include: {
        resultingModel: true,
        submissions: true,
        aggregationJobs: true,
      },
    });

    if (!round) return res.status(404).json({ error: 'Round not found' });

    // Strict deletion rules
    if (round.status === 'COMPLETED') {
      return res.status(400).json({
        error: 'COMPLETED rounds cannot be deleted to preserve academic research history and reproducibility.',
      });
    }

    if (round.status === 'ACTIVE' || round.status === 'AGGREGATING') {
      return res.status(400).json({
        error: `Round is currently '${round.status}'. You must cancel the round before deleting it.`,
      });
    }

    if (round.resultingModelId) {
      return res.status(400).json({
        error: 'This round resulted in a published global model and cannot be deleted.',
      });
    }

    if (!['CREATED', 'CANCELLED', 'FAILED'].includes(round.status)) {
      return res.status(400).json({
        error: `Cannot delete round with status '${round.status}'.`,
      });
    }

    // Foreign-key-safe transaction
    await prisma.$transaction(async (tx) => {
      // 1. Delete aggregation weights & jobs
      const jobs = await tx.aggregationJob.findMany({ where: { roundId }, select: { id: true } });
      const jobIds = jobs.map((j) => j.id);
      if (jobIds.length > 0) {
        await tx.aggregationWeight.deleteMany({ where: { aggregationJobId: { in: jobIds } } });
        await tx.aggregationJob.deleteMany({ where: { roundId } });
      }

      // 2. Delete ARC scores for this round
      await tx.aRCScore.deleteMany({ where: { roundId } });

      // 3. Delete submissions for this round
      await tx.clientSubmission.deleteMany({ where: { roundId } });

      // 4. Delete the round itself
      await tx.trainingRound.delete({ where: { id: roundId } });

      // 5. Create audit log
      await tx.auditLog.create({
        data: {
          userId: req.user.id,
          action: 'DELETE_ROUND',
          resourceType: 'TrainingRound',
          resourceId: roundId,
          detailsJson: {
            roundNumber: round.roundNumber,
            status: round.status,
            name: round.name,
          },
          ipAddress: req.ip,
        },
      });
    });

    res.json({
      message: `Round #${round.roundNumber} (${round.name || roundId}) deleted successfully.`,
      roundId,
    });
  } catch (err) {
    next(err);
  }
}

/**
 * GET /api/admin/rounds/:roundId/submissions
 * List all submissions for a specific round.
 */
async function getSubmissions(req, res, next) {
  try {
    const { roundId } = req.params;
    const submissions = await prisma.clientSubmission.findMany({
      where: { roundId },
      include: {
        client: { select: { clientId: true, name: true } },
        arcScores: { orderBy: { calculatedAt: 'desc' }, take: 1 },
      },
      orderBy: { submittedAt: 'desc' },
    });

    res.json({ submissions });
  } catch (err) {
    next(err);
  }
}

/**
 * PATCH /api/admin/submissions/:submissionId/review
 * Accept or reject a client submission.
 */
async function reviewSubmission(req, res, next) {
  try {
    const { submissionId } = req.params;
    const { status, validationMessage } = req.body;

    if (!['ACCEPTED', 'REJECTED'].includes(status)) {
      return res.status(400).json({ error: 'Status must be ACCEPTED or REJECTED' });
    }

    const submission = await prisma.clientSubmission.update({
      where: { id: submissionId },
      data: {
        status,
        validationMessage,
        reviewedAt: new Date(),
        reviewedBy: req.user.id,
      },
    });

    res.json({ submission });
  } catch (err) {
    next(err);
  }
}

/**
 * GET /api/admin/submissions
 * List all submissions fleet-wide (optionally filtered by roundId).
 */
async function getAllSubmissions(req, res, next) {
  try {
    const { roundId, status } = req.query;
    const where = {};
    if (roundId) where.roundId = roundId;
    if (status) where.status = status;

    const submissions = await prisma.clientSubmission.findMany({
      where,
      include: {
        client: { select: { clientId: true, name: true, status: true } },
        round: { select: { roundNumber: true, status: true } },
        arcScores: { orderBy: { calculatedAt: 'desc' }, take: 1 },
      },
      orderBy: { submittedAt: 'desc' },
    });

    res.json({ submissions });
  } catch (err) {
    next(err);
  }
}

/**
 * GET /api/admin/analytics
 * Returns historical analytics across rounds for convergence & distribution charts.
 */
async function getAnalytics(req, res, next) {
  try {
    const rounds = await prisma.trainingRound.findMany({
      orderBy: { roundNumber: 'asc' },
      include: {
        baseModel: { select: { version: true, baselineAccuracy: true, baselineLoss: true } },
        resultingModel: { select: { version: true, baselineAccuracy: true, baselineLoss: true } },
        submissions: {
          select: {
            id: true,
            clientId: true,
            localAccuracy: true,
            localLoss: true,
            samplesTrained: true,
            arcScore: true,
          },
        },
        arcScores: {
          select: {
            relevanceScore: true,
            resourceScore: true,
            communicationScore: true,
            overallArcScore: true,
          },
        },
      },
    });

    const roundData = rounds.map((r) => {
      const accuracies = r.submissions.map((s) => s.localAccuracy).filter(Boolean);
      const losses = r.submissions.map((s) => s.localLoss).filter(Boolean);
      const arcs = r.arcScores.map((a) => a.overallArcScore).filter(Boolean);

      return {
        roundNumber: r.roundNumber,
        status: r.status,
        clientCount: r.submissions.length,
        avgLocalAccuracy: accuracies.length ? parseFloat((accuracies.reduce((a, b) => a + b, 0) / accuracies.length).toFixed(2)) : null,
        avgLocalLoss: losses.length ? parseFloat((losses.reduce((a, b) => a + b, 0) / losses.length).toFixed(4)) : null,
        avgArcScore: arcs.length ? parseFloat((arcs.reduce((a, b) => a + b, 0) / arcs.length).toFixed(2)) : null,
        globalAccuracy: r.resultingModel?.baselineAccuracy || r.baseModel?.baselineAccuracy || null,
        globalLoss: r.resultingModel?.baselineLoss || r.baseModel?.baselineLoss || null,
      };
    });

    // Prepend Round 0 as baseline initial model reference if present
    const initialModel = await prisma.globalModel.findFirst({ where: { version: 'initial-v1' } });
    const baselinePoint = initialModel ? {
      roundNumber: 0,
      status: 'BASELINE',
      clientCount: 0,
      avgLocalAccuracy: null,
      avgLocalLoss: null,
      avgArcScore: null,
      globalAccuracy: initialModel.baselineAccuracy || 98.5,
      globalLoss: initialModel.baselineLoss || 0.12,
    } : null;

    const fullRoundData = baselinePoint ? [baselinePoint, ...roundData] : roundData;

    res.json({ roundData: fullRoundData });
  } catch (err) {
    next(err);
  }
}

/**
 * GET /api/admin/audit-logs
 * List recent administrative action audit logs.
 */
async function getAuditLogs(req, res, next) {
  try {
    const logs = await prisma.auditLog.findMany({
      take: 50,
      orderBy: { timestamp: 'desc' },
      include: { user: { select: { name: true, email: true, role: true } } },
    });
    res.json({ logs });
  } catch (err) {
    next(err);
  }
}

/**
 * GET /api/admin/system-health
 * Proxies health status of Node backend, PostgreSQL DB, and Python ML Service.
 * Does not expose internal endpoints to the browser.
 */
async function getSystemHealth(req, res, next) {
  try {
    const t0 = Date.now();
    let dbStatus = 'disconnected';
    let dbLatencyMs = null;
    try {
      await prisma.$queryRaw`SELECT 1`;
      dbLatencyMs = Date.now() - t0;
      dbStatus = 'connected';
    } catch (dbErr) {
      dbStatus = 'error';
    }

    let mlStatus = 'offline';
    let mlLatencyMs = null;
    let mlDetails = null;
    let mlError = null;
    try {
      const mlStart = Date.now();
      const mlData = await mlServiceClient.getHealth();
      mlLatencyMs = Date.now() - mlStart;
      mlStatus = mlData?.status === 'ok' ? 'online' : 'degraded';
      mlDetails = {
        service: mlData?.service,
        version: mlData?.version,
        modelLoaded: mlData?.model_loaded,
        pytorchVersion: mlData?.pytorch_version,
        modelParams: mlData?.model_params,
      };
    } catch (mlErr) {
      mlStatus = 'offline';
      mlError = mlErr.message || 'Python ML Service unreachable on port 8000';
    }

    const mem = process.memoryUsage();

    res.json({
      timestamp: new Date().toISOString(),
      nodeBackend: {
        status: 'online',
        port: config.port || 5000,
        uptimeSeconds: Math.floor(process.uptime()),
        memoryRssMb: Math.round(mem.rss / (1024 * 1024)),
      },
      database: {
        status: dbStatus,
        latencyMs: dbLatencyMs,
      },
      pythonMlService: {
        status: mlStatus,
        port: 8000,
        latencyMs: mlLatencyMs,
        details: mlDetails,
        error: mlError,
      },
    });
  } catch (err) {
    next(err);
  }
}

/**
 * POST /api/admin/dataset-audit
 * Inspect client datasets and report missing classes and class distribution deficits.
 * Body: { clientIds: string[] }
 */
async function checkDatasets(req, res, next) {
  try {
    const { clientIds = [] } = req.body;
    const auditResult = auditClientDatasets(clientIds);
    res.json(auditResult);
  } catch (err) {
    next(err);
  }
}

/**
 * GET /api/admin/rounds/:roundId/dataset-audit
 * Inspect datasets of the target clients participating in an existing round.
 */
async function getRoundDatasetAudit(req, res, next) {
  try {
    const { roundId } = req.params;
    const round = await prisma.trainingRound.findUnique({
      where: { id: roundId },
      select: { id: true, roundNumber: true, targetClients: true },
    });
    if (!round) {
      return res.status(404).json({ error: 'Round not found' });
    }
    const auditResult = auditClientDatasets(round.targetClients || []);
    res.json({ roundId: round.id, roundNumber: round.roundNumber, ...auditResult });
  } catch (err) {
    next(err);
  }
}

module.exports = {
  getDashboard,
  getClients,
  updateClientStatus,
  getRounds,
  getRoundDetails,
  createRound,
  updateRound,
  startRound,
  cancelRound,
  deleteRound,
  checkRoundDependencies,
  archiveRound,
  getSubmissions,
  reviewSubmission,
  getAllSubmissions,
  getAnalytics,
  getAuditLogs,
  getSystemHealth,
  checkDatasets,
  getRoundDatasetAudit,
};
