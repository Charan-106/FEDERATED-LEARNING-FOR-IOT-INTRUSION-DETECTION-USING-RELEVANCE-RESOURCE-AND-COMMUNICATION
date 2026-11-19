// ARC Controller - Score retrieval, dynamic configuration, and recalculation
// Handles the ARC (Adaptive Resource-aware Contribution) scoring framework (v1.0)

const prisma = require('../config/database');
const mlServiceClient = require('../services/mlServiceClient');

/**
 * GET /api/arc/scores?roundId=...
 * Get all ARC scores for a training round or fleet-wide.
 */
async function getScores(req, res, next) {
  try {
    const { roundId } = req.query;

    const where = {};
    if (roundId) where.roundId = roundId;

    const scores = await prisma.aRCScore.findMany({
      where,
      include: {
        client: { select: { clientId: true, name: true, hardwareSpecs: true, status: true } },
        submission: {
          select: {
            id: true,
            status: true,
            submittedAt: true,
            localAccuracy: true,
            localLoss: true,
            samplesTrained: true,
          },
        },
        round: { select: { roundNumber: true, status: true } },
      },
      orderBy: { calculatedAt: 'desc' },
    });

    res.json({ scores });
  } catch (err) {
    next(err);
  }
}

/**
 * GET /api/arc/config
 * Get the currently active ARC configuration (formula weights).
 */
async function getConfig(req, res, next) {
  try {
    const config = await prisma.aRCConfig.findFirst({ where: { isActive: true } });
    if (!config) {
      return res.status(404).json({ error: 'No active ARC configuration found' });
    }
    res.json({ config });
  } catch (err) {
    next(err);
  }
}

/**
 * GET /api/arc/configs
 * Retrieve configuration history (all versions created).
 */
async function getConfigHistory(req, res, next) {
  try {
    const configs = await prisma.aRCConfig.findMany({
      orderBy: { createdAt: 'desc' },
    });
    res.json({ configs });
  } catch (err) {
    next(err);
  }
}

/**
 * PUT /api/arc/config
 * Update ARC formula weights. Validates that lambda weights sum to 1.0.
 * Persists new version, deactivates older versions, creates AuditLog entry,
 * and optionally triggers recalculation of pending submissions.
 */
async function updateConfig(req, res, next) {
  try {
    const {
      lambdaRelevance,
      lambdaResource,
      lambdaCommunication,
      weightQuality,
      weightDiversity,
      weightMemory,
      weightProcessing,
      weightCapacity,
      weightBandwidth,
      weightLatency,
      weightReliability,
      recalculatePending,
    } = req.body;

    // Validate top-level lambda weights sum to 1.0 (allow small floating point tolerance)
    if (lambdaRelevance !== undefined && lambdaResource !== undefined && lambdaCommunication !== undefined) {
      const lambdaSum = parseFloat(lambdaRelevance) + parseFloat(lambdaResource) + parseFloat(lambdaCommunication);
      if (Math.abs(lambdaSum - 1.0) > 0.005) {
        return res.status(400).json({
          error: 'Validation failed',
          message: `Lambda weights must sum to 1.0, got ${lambdaSum.toFixed(4)}`,
        });
      }
    }

    // Previous active config
    const currentConfig = await prisma.aRCConfig.findFirst({
      where: { isActive: true },
      orderBy: { createdAt: 'desc' },
    });

    // Count existing configs to increment version
    const totalConfigs = await prisma.aRCConfig.count();
    const newVersion = `1.${totalConfigs}`;

    // Deactivate current active configs
    await prisma.aRCConfig.updateMany({ where: { isActive: true }, data: { isActive: false } });

    // Create new active config record
    const newConfig = await prisma.aRCConfig.create({
      data: {
        version: newVersion,
        isActive: true,
        lambdaRelevance: lambdaRelevance !== undefined ? parseFloat(lambdaRelevance) : (currentConfig?.lambdaRelevance ?? 0.40),
        lambdaResource: lambdaResource !== undefined ? parseFloat(lambdaResource) : (currentConfig?.lambdaResource ?? 0.30),
        lambdaCommunication: lambdaCommunication !== undefined ? parseFloat(lambdaCommunication) : (currentConfig?.lambdaCommunication ?? 0.30),
        weightQuality: weightQuality !== undefined ? parseFloat(weightQuality) : (currentConfig?.weightQuality ?? 0.60),
        weightDiversity: weightDiversity !== undefined ? parseFloat(weightDiversity) : (currentConfig?.weightDiversity ?? 0.40),
        weightMemory: weightMemory !== undefined ? parseFloat(weightMemory) : (currentConfig?.weightMemory ?? 0.40),
        weightProcessing: weightProcessing !== undefined ? parseFloat(weightProcessing) : (currentConfig?.weightProcessing ?? 0.40),
        weightCapacity: weightCapacity !== undefined ? parseFloat(weightCapacity) : (currentConfig?.weightCapacity ?? 0.20),
        weightBandwidth: weightBandwidth !== undefined ? parseFloat(weightBandwidth) : (currentConfig?.weightBandwidth ?? 0.50),
        weightLatency: weightLatency !== undefined ? parseFloat(weightLatency) : (currentConfig?.weightLatency ?? 0.30),
        weightReliability: weightReliability !== undefined ? parseFloat(weightReliability) : (currentConfig?.weightReliability ?? 0.20),
      },
    });

    // Audit log the weight update
    await prisma.auditLog.create({
      data: {
        userId: req.user?.id || null,
        action: 'UPDATE_ARC_CONFIG',
        resourceType: 'ARCConfig',
        resourceId: newConfig.id,
        detailsJson: {
          previousVersion: currentConfig?.version || '1.0',
          newVersion,
          previousWeights: currentConfig ? {
            lambdaRelevance: currentConfig.lambdaRelevance,
            lambdaResource: currentConfig.lambdaResource,
            lambdaCommunication: currentConfig.lambdaCommunication,
          } : null,
          newWeights: {
            lambdaRelevance: newConfig.lambdaRelevance,
            lambdaResource: newConfig.lambdaResource,
            lambdaCommunication: newConfig.lambdaCommunication,
          },
        },
        ipAddress: req.ip || null,
      },
    });

    let recalculationResult = null;
    if (recalculatePending) {
      recalculationResult = await executeRecalculation({ configRecord: newConfig, userId: req.user?.id });
    }

    res.json({
      config: newConfig,
      recalculation: recalculationResult,
      message: `ARC configuration updated to version ${newVersion} successfully`,
    });
  } catch (err) {
    next(err);
  }
}

/**
 * POST /api/arc/recalculate
 * Explicitly triggers recalculation of ARC scores for submissions using the current active weights.
 */
async function recalculateScores(req, res, next) {
  try {
    const { roundId, submissionIds } = req.body;
    const activeConfig = await prisma.aRCConfig.findFirst({ where: { isActive: true } });
    if (!activeConfig) {
      return res.status(400).json({ error: 'No active ARC configuration found' });
    }

    const result = await executeRecalculation({
      configRecord: activeConfig,
      roundId,
      submissionIds,
      userId: req.user?.id,
    });

    res.json({
      message: `Successfully recalculated ARC scores for ${result.recalculatedCount} submission(s)`,
      ...result,
    });
  } catch (err) {
    next(err);
  }
}

/**
 * Internal helper to recalculate scores for target submissions
 */
async function executeRecalculation({ configRecord, roundId, submissionIds, userId }) {
  const where = { status: 'PENDING' };
  if (roundId) where.roundId = roundId;
  if (submissionIds && submissionIds.length > 0) where.id = { in: submissionIds };

  const submissions = await prisma.clientSubmission.findMany({
    where,
    include: {
      arcScores: { orderBy: { calculatedAt: 'desc' }, take: 1 },
      client: true,
    },
  });

  const recalculated = [];

  for (const sub of submissions) {
    const existingArc = sub.arcScores[0];
    const telemetry = existingArc?.rawTelemetryJson || {
      cpu_usage_pct: 25,
      ram_usage_gb: 4.0,
      total_ram_gb: 16.0,
      available_ram_gb: 12.0,
      throughput_samples_per_sec: 500,
      bandwidth_mbps: 28.5,
      latency_ms: 38.0,
      reliability_pct: 99.5,
    };

    try {
      const arcResult = await mlServiceClient.calculateARC({
        submissionId: sub.id,
        clientId: sub.clientId,
        roundId: sub.roundId,
        localAccuracy: sub.localAccuracy || 0,
        localLoss: sub.localLoss || 0,
        samplesTrained: sub.samplesTrained || 0,
        telemetry,
        arcConfig: {
          lambdaRelevance: configRecord.lambdaRelevance,
          lambdaResource: configRecord.lambdaResource,
          lambdaCommunication: configRecord.lambdaCommunication,
          weightQuality: configRecord.weightQuality,
          weightDiversity: configRecord.weightDiversity,
          weightMemory: configRecord.weightMemory,
          weightProcessing: configRecord.weightProcessing,
          weightCapacity: configRecord.weightCapacity,
          weightBandwidth: configRecord.weightBandwidth,
          weightLatency: configRecord.weightLatency,
          weightReliability: configRecord.weightReliability,
        },
      });

      if (arcResult && arcResult.overall_arc_score !== undefined) {
        // Record new score in ARCScore table
        const newScore = await prisma.aRCScore.create({
          data: {
            roundId: sub.roundId,
            submissionId: sub.id,
            clientId: sub.clientId,
            relevanceScore: arcResult.relevance_score,
            resourceScore: arcResult.resource_score,
            communicationScore: arcResult.communication_score,
            overallArcScore: arcResult.overall_arc_score,
            qualityScore: arcResult.quality_score,
            diversityScore: arcResult.diversity_score,
            memoryScore: arcResult.memory_score,
            processingScore: arcResult.processing_score,
            capacityScore: arcResult.capacity_score,
            bandwidthScore: arcResult.bandwidth_score,
            latencyScore: arcResult.latency_score,
            reliabilityScore: arcResult.reliability_score,
            formulaVersion: configRecord.version,
            weightsUsedJson: arcResult.weights_used,
            rawTelemetryJson: telemetry,
          },
        });

        // Update cached ARC score on submission
        await prisma.clientSubmission.update({
          where: { id: sub.id },
          data: { arcScore: arcResult.overall_arc_score },
        });

        recalculated.push({
          submissionId: sub.id,
          clientId: sub.clientId,
          previousScore: existingArc?.overallArcScore || sub.arcScore,
          newScore: arcResult.overall_arc_score,
          delta: roundToFour(arcResult.overall_arc_score - (existingArc?.overallArcScore || sub.arcScore || 0)),
          relevance: arcResult.relevance_score,
          resource: arcResult.resource_score,
          communication: arcResult.communication_score,
        });
      }
    } catch (err) {
      console.error(`[recalculateScores] Failed for sub ${sub.id}:`, err.message);
    }
  }

  // Create audit log
  await prisma.auditLog.create({
    data: {
      userId: userId || null,
      action: 'RECALCULATE_ARC',
      resourceType: 'ARCScore',
      detailsJson: {
        recalculatedCount: recalculated.length,
        configVersion: configRecord.version,
        submissions: recalculated.map(r => ({ id: r.submissionId, prev: r.previousScore, next: r.newScore })),
      },
    },
  });

  return {
    recalculatedCount: recalculated.length,
    configVersion: configRecord.version,
    results: recalculated,
  };
}

function roundToFour(num) {
  return Math.round((num + Number.EPSILON) * 10000) / 10000;
}

module.exports = {
  getScores,
  getConfig,
  getConfigHistory,
  updateConfig,
  recalculateScores,
};
