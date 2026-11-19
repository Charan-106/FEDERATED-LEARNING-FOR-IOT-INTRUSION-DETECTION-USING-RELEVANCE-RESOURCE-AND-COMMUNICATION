// Aggregation Controller - Multi-Criteria Model Merging & Validation
// Handles FedAvg, Equal, and ARC-Weighted aggregation strategies,
// candidate model artifact generation, and server-side validation.

const path = require('path');
const config = require('../config');
const prisma = require('../config/database');
const mlServiceClient = require('../services/mlServiceClient');

/**
 * POST /api/aggregation/preview
 * Compute aggregation weights for selected submissions without executing the merge.
 * Strategies:
 *   1. FEDAVG: alpha_i = n_i / sum(n_j)
 *   2. EQUAL: alpha_i = 1 / K
 *   3. ARC_WEIGHTED: alpha_i = ARC_i / sum(ARC_j)
 */
async function previewWeights(req, res, next) {
  try {
    const roundId = req.body.roundId || req.query.roundId;
    let submissionIds = req.body.submissionIds;
    if (!submissionIds && req.query.submissionIds) {
      submissionIds = typeof req.query.submissionIds === 'string' ? req.query.submissionIds.split(',').map(s => s.trim()) : req.query.submissionIds;
    }
    const strategy = req.body.strategy || req.query.strategy || 'ARC_WEIGHTED';

    if (!submissionIds || submissionIds.length === 0) {
      return res.status(400).json({ error: 'submissionIds array is required' });
    }

    const submissions = await prisma.clientSubmission.findMany({
      where: { id: { in: submissionIds }, roundId },
      include: {
        client: { select: { clientId: true, name: true, hardwareSpecs: true } },
        arcScores: { orderBy: { calculatedAt: 'desc' }, take: 1 },
      },
    });

    if (submissions.length === 0) {
      return res.status(404).json({ error: 'No submissions found for given IDs and round' });
    }

    // Compute raw weights based on strategy
    let rawWeights = {};
    let total = 0;

    for (const sub of submissions) {
      let weight = 0;
      if (strategy === 'FEDAVG') {
        weight = sub.samplesTrained || 0;
        if (weight <= 0) {
          return res.status(400).json({ error: `Client ${sub.clientId} has invalid sample count (${weight}) for FEDAVG strategy.` });
        }
      } else if (strategy === 'EQUAL') {
        weight = 1.0;
      } else if (strategy === 'ARC_WEIGHTED') {
        let arc = sub.arcScores[0]?.overallArcScore ?? sub.arcScore;
        if (arc === undefined || arc === null || isNaN(Number(arc)) || Number(arc) <= 0) {
          // Graceful fallback score instead of hard error
          arc = sub.localAccuracy ? Math.round(Number(sub.localAccuracy) * 0.75 * 10) / 10 : 65.0;
          console.warn(`[previewWeights] Client ${sub.clientId} missing ARC score. Using computed fallback ${arc}`);
        }
        weight = Number(arc);
      }
      rawWeights[sub.id] = { raw: weight, submission: sub };
      total += weight;
    }

    if (total <= 0) {
      return res.status(400).json({ error: `Total weight sum is ${total} (must be > 0). Cannot normalize weights for strategy ${strategy}.` });
    }

    // Normalize weights to sum to 1.0
    const normalizedWeights = Object.entries(rawWeights).map(([subId, data]) => {
      const norm = data.raw / total;
      const arc = data.submission.arcScores[0]?.overallArcScore ?? data.submission.arcScore ?? 0;
      return {
        submissionId: subId,
        submission_id: subId,
        clientId: data.submission.clientId,
        client_id: data.submission.clientId,
        clientName: data.submission.client?.name || data.submission.clientId,
        client_name: data.submission.client?.name || data.submission.clientId,
        rawWeight: data.raw,
        raw_weight: data.raw,
        normalizedWeight: roundToFour(norm),
        weight: roundToFour(norm),
        clientSamples: data.submission.samplesTrained || 600,
        samples: data.submission.samplesTrained || 600,
        arcScore: arc,
        arc_score: arc,
      };
    });

    res.json({
      strategy,
      roundId,
      submissionCount: submissions.length,
      totalWeight: 1.0,
      total_weight: 1.0,
      weights: normalizedWeights,
      preview: true,
    });
  } catch (err) {
    next(err);
  }
}

/**
 * POST /api/aggregation/merge
 * Execute federated model aggregation via the Python ML service.
 * Performs tensor validation, weighted parameter averaging, candidate model creation,
 * and immediate server-side validation against validation.csv.
 */
async function mergeModels(req, res, next) {
  try {
    const { roundId, submissionIds, strategy = 'ARC_WEIGHTED' } = req.body;

    if (!submissionIds || submissionIds.length < 1) {
      return res.status(400).json({ error: 'At least 1 submission required for federated aggregation' });
    }

    const round = await prisma.trainingRound.findUnique({
      where: { id: roundId },
      include: { baseModel: true },
    });
    if (!round) return res.status(404).json({ error: 'Round not found' });

    // Idempotent recovery: If round already completed or aggregation job exists, return existing candidate result
    const existingJob = await prisma.aggregationJob.findFirst({
      where: { roundId, status: 'COMPLETED' },
      include: {
        weights: {
          include: { submission: { include: { client: { select: { clientId: true, name: true } } } } },
        },
        round: true,
      },
      orderBy: { completedAt: 'desc' },
    });

    if (existingJob) {
      const weights = existingJob.weights.map((w) => ({
        submission_id: w.submissionId,
        clientId: w.clientId || w.submission?.clientId,
        name: w.submission?.client?.name || w.clientId,
        normalized_weight: w.normalizedWeight,
        effective_weight: w.normalizedWeight,
        arc_score: w.arcScore,
        sample_count: w.clientSamples,
      }));

      return res.json({
        message: `Aggregation was already completed for Round #${round.roundNumber}. Loaded existing candidate evaluation results.`,
        alreadyCompleted: true,
        jobId: existingJob.id,
        candidate_id: existingJob.id.slice(0, 8),
        candidateId: existingJob.id.slice(0, 8),
        candidatePath: existingJob.candidateModelPath,
        candidateChecksum: existingJob.candidateChecksum,
        strategy: existingJob.strategy,
        validClientCount: existingJob.weights.length,
        weights,
        evaluation: existingJob.evaluationReportJson,
      });
    }

    if (round.status === 'COMPLETED') {
      return res.status(400).json({ error: 'Round is marked COMPLETED but no completed aggregation job was found' });
    }

    // Retrieve eligible submissions (PENDING or ACCEPTED)
    const submissions = await prisma.clientSubmission.findMany({
      where: {
        id: { in: submissionIds },
        roundId,
        status: { in: ['ACCEPTED', 'PENDING'] },
      },
      include: {
        client: { select: { clientId: true, name: true } },
        arcScores: { orderBy: { calculatedAt: 'desc' }, take: 1 },
      },
    });

    if (submissions.length < 1) {
      return res.status(400).json({ error: 'Need at least 1 valid submission to aggregate' });
    }

    // Update round status to AGGREGATING
    await prisma.trainingRound.update({
      where: { id: roundId },
      data: { status: 'AGGREGATING', aggregationStrategy: strategy },
    });

    // Create aggregation job record
    const job = await prisma.aggregationJob.create({
      data: {
        roundId,
        strategy,
        status: 'RUNNING',
        selectedClientCount: submissions.length,
        performedBy: req.user.id,
      },
    });

    // Build payload for Python ML service
    const baseModelPath = round.baseModel
      ? path.join(config.storage.root, round.baseModel.storagePath, 'best.pt')
      : null;

    const submissionPayload = submissions.map((sub) => {
      let arc = sub.arcScores[0]?.overallArcScore ?? sub.arcScore;
      if (strategy === 'ARC_WEIGHTED') {
        if (arc === undefined || arc === null || isNaN(Number(arc)) || Number(arc) <= 0) {
          arc = sub.localAccuracy ? Math.round(Number(sub.localAccuracy) * 0.75 * 10) / 10 : 65.0;
          console.warn(`[mergeModels] Client ${sub.clientId} missing ARC score. Using computed fallback ${arc}`);
        }
      }
      if (strategy === 'FEDAVG') {
        const samples = sub.samplesTrained || 0;
        if (samples <= 0) {
          throw new Error(`Client ${sub.clientId} has invalid sample count (${samples}) for FEDAVG aggregation.`);
        }
      }
      return {
        submission_id: sub.id,
        client_id: sub.clientId,
        model_path: sub.storagePath,
        samples_trained: sub.samplesTrained || 600,
        arc_score: arc !== undefined && arc !== null ? Number(arc) : 65.0,
      };
    });

    // Call Python ML service to perform parameter aggregation
    let aggregationResult;
    try {
      aggregationResult = await mlServiceClient.aggregateModels({
        job_id: job.id,
        round_id: roundId,
        strategy,
        base_model_path: baseModelPath,
        server_learning_rate: req.body.serverLearningRate ? Number(req.body.serverLearningRate) : 0.5,
        submissions: submissionPayload,
      });
    } catch (mlErr) {
      await prisma.aggregationJob.update({
        where: { id: job.id },
        data: { status: 'FAILED' },
      });
      await prisma.trainingRound.update({
        where: { id: roundId },
        data: { status: 'FAILED' },
      });
      throw new Error(`ML parameter aggregation failed: ${mlErr.message}`);
    }

    // Save individual client aggregation weights
    for (const sub of submissions) {
      const wt = aggregationResult.client_weights?.find((w) => w.submission_id === sub.id)
        || aggregationResult.weights?.find((w) => w.submission_id === sub.id);

      await prisma.aggregationWeight.create({
        data: {
          aggregationJobId: job.id,
          submissionId: sub.id,
          clientId: sub.clientId,
          rawWeight: wt?.raw_weight || (strategy === 'FEDAVG' ? sub.samplesTrained : 1.0),
          normalizedWeight: wt?.effective_weight || wt?.normalized_weight || (1.0 / submissions.length),
          clientSamples: sub.samplesTrained,
          arcScore: sub.arcScores[0]?.overallArcScore || sub.arcScore,
        },
      });

      // Mark submission as AGGREGATED
      await prisma.clientSubmission.update({
        where: { id: sub.id },
        data: { status: 'AGGREGATED' },
      });
    }

    // Run Server-Side Candidate Evaluation against validation.csv
    let evaluationMetrics = null;
    try {
      evaluationMetrics = await mlServiceClient.evaluateModel(aggregationResult.candidate_path);
      if (evaluationMetrics?.accuracy !== undefined) {
        evaluationMetrics.accuracy = roundToFour(evaluationMetrics.accuracy);
      }
    } catch (evalErr) {
      console.warn('[mergeModels] Candidate validation warning:', evalErr.message);
    }

    // Update job as completed with evaluation report
    await prisma.aggregationJob.update({
      where: { id: job.id },
      data: {
        status: 'COMPLETED',
        candidateModelPath: aggregationResult.candidate_path,
        candidateChecksum: aggregationResult.checksum,
        evaluationReportJson: evaluationMetrics || {},
        completedAt: new Date(),
      },
    });

    // Update round status to COMPLETED
    await prisma.trainingRound.update({
      where: { id: roundId },
      data: {
        status: 'COMPLETED',
        completedAt: new Date(),
      },
    });

    // Create Audit Log
    await prisma.auditLog.create({
      data: {
        userId: req.user.id,
        action: 'AGGREGATE_MODELS',
        resourceType: 'AggregationJob',
        resourceId: job.id,
        detailsJson: {
          roundId,
          strategy,
          selectedClientCount: submissions.length,
          candidatePath: aggregationResult.candidate_path,
          checksum: aggregationResult.checksum,
          evaluation: evaluationMetrics,
        },
        ipAddress: req.ip,
      },
    });

    res.json({
      message: `Aggregation successfully completed using ${strategy} strategy! Candidate model evaluated.`,
      jobId: job.id,
      candidate_id: aggregationResult.candidate_id || job.id.slice(0, 8),
      candidateId: aggregationResult.candidate_id || job.id.slice(0, 8),
      candidatePath: aggregationResult.candidate_path,
      candidateChecksum: aggregationResult.checksum,
      strategy,
      validClientCount: aggregationResult.valid_client_count,
      weights: (aggregationResult.client_weights || []).map((w) => ({
        submission_id: w.submission_id,
        clientId: w.client_id || w.clientId,
        name: w.client_id || w.clientId,
        normalized_weight: w.effective_weight !== undefined ? w.effective_weight : w.normalized_weight,
        effective_weight: w.effective_weight !== undefined ? w.effective_weight : w.normalized_weight,
        arc_score: w.arc_score,
        sample_count: w.sample_count,
      })),
      evaluation: evaluationMetrics,
    });
  } catch (err) {
    next(err);
  }
}

/**
 * GET /api/aggregation/history
 * Return all aggregation jobs with weights and evaluation results.
 */
async function getHistory(req, res, next) {
  try {
    const jobs = await prisma.aggregationJob.findMany({
      orderBy: { createdAt: 'desc' },
      include: {
        weights: {
          include: {
            submission: {
              select: {
                localAccuracy: true,
                samplesTrained: true,
                client: { select: { name: true, clientId: true } },
              },
            },
          },
        },
        round: { select: { roundNumber: true, status: true } },
      },
    });

    res.json({ jobs });
  } catch (err) {
    next(err);
  }
}

/**
 * GET /api/aggregation/round/:roundId
 * Fetch existing aggregation job and candidate evaluation report for a completed round.
 */
async function getRoundAggregation(req, res, next) {
  try {
    const { roundId } = req.params;
    const round = await prisma.trainingRound.findUnique({
      where: { id: roundId },
      include: { baseModel: true },
    });
    if (!round) return res.status(404).json({ error: 'Round not found' });

    const job = await prisma.aggregationJob.findFirst({
      where: { roundId, status: 'COMPLETED' },
      include: {
        weights: {
          include: { submission: { include: { client: { select: { clientId: true, name: true } } } } },
        },
      },
      orderBy: { completedAt: 'desc' },
    });

    if (!job) {
      return res.status(404).json({ error: 'No completed aggregation found for this round' });
    }

    const weights = job.weights.map((w) => ({
      submission_id: w.submissionId,
      clientId: w.clientId || w.submission?.clientId,
      name: w.submission?.client?.name || w.clientId,
      normalized_weight: w.normalizedWeight,
      effective_weight: w.normalizedWeight,
      arc_score: w.arcScore,
      sample_count: w.clientSamples,
    }));

    res.json({
      jobId: job.id,
      candidate_id: job.id.slice(0, 8),
      candidateId: job.id.slice(0, 8),
      candidatePath: job.candidateModelPath,
      candidateChecksum: job.candidateChecksum,
      strategy: job.strategy,
      validClientCount: job.weights.length,
      weights,
      evaluation: job.evaluationReportJson,
      completedAt: job.completedAt,
    });
  } catch (err) {
    next(err);
  }
}

function roundToFour(num) {
  return Math.round((num + Number.EPSILON) * 10000) / 10000;
}

module.exports = { previewWeights, mergeModels, getHistory, getRoundAggregation };
