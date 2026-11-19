// Round Lifecycle & Client Eligibility Service
// Enforces strict federated round resolution across all clients and lifecycle stages.
// Rules:
// 1. Never selects older stale rounds when a newer eligible round exists.
// 2. Selects latest eligible active round based on round number, creation time, and client participation.
// 3. Clients who already submitted are marked as hasSubmitted = true and prevented from re-training.
// 4. Other assigned clients who have not yet submitted see the same round as ready to train.
// 5. Completed rounds (status === 'COMPLETED') appear in historical archive.
// 6. When latest round is COMPLETED, no client is sent to older stale active rounds.

const prisma = require('../config/database');

/**
 * Checks if a client is targeted for a round.
 * Null or empty targetClients array means all registered clients are targeted.
 */
function isClientTargeted(round, clientId) {
  if (!round) return false;
  const targets = round.targetClients;
  if (!targets) return true;
  if (!Array.isArray(targets)) return true;
  if (targets.length === 0) return true;
  return targets.includes(clientId);
}

/**
 * Resolves the current eligible training round for a client.
 * @param {string} clientId - The unique client identifier (e.g. 'client-001')
 * @param {string|null} requestedRoundId - Optional explicit round ID
 * @returns {Promise<{ round: object|null, isTargeted: boolean, hasSubmitted: boolean, submission: object|null, reason: string }>}
 */
async function resolveEligibleRoundForClient(clientId, requestedRoundId = null) {
  // Case 1: Explicit round requested by ID
  if (requestedRoundId) {
    const round = await prisma.trainingRound.findUnique({
      where: { id: requestedRoundId },
      include: { baseModel: true },
    });

    if (!round) {
      return { round: null, isTargeted: false, hasSubmitted: false, submission: null, reason: 'ROUND_NOT_FOUND' };
    }

    const targeted = isClientTargeted(round, clientId);
    const submission = await prisma.clientSubmission.findFirst({
      where: { roundId: round.id, clientId },
      orderBy: { submittedAt: 'desc' },
      include: { arcScores: { orderBy: { calculatedAt: 'desc' }, take: 1 } },
    });

    return {
      round,
      isTargeted: targeted,
      hasSubmitted: Boolean(submission),
      submission,
      reason: 'EXPLICIT_SELECTION',
    };
  }

  // Case 2: Automatic resolution
  // First, find the latest round overall in the platform (by roundNumber DESC)
  const latestRoundOverall = await prisma.trainingRound.findFirst({
    orderBy: { roundNumber: 'desc' },
    include: {
      baseModel: true,
      submissions: {
        where: { clientId },
        orderBy: { submittedAt: 'desc' },
        take: 1,
        include: { arcScores: { orderBy: { calculatedAt: 'desc' }, take: 1 } },
      },
    },
  });

  if (!latestRoundOverall) {
    return { round: null, isTargeted: false, hasSubmitted: false, submission: null, reason: 'NO_ROUNDS_EXIST' };
  }

  // If the latest round is ACTIVE:
  if (latestRoundOverall.status === 'ACTIVE') {
    const isTargeted = isClientTargeted(latestRoundOverall, clientId);
    const hasSubmitted = latestRoundOverall.submissions.length > 0;

    if (isTargeted) {
      return {
        round: latestRoundOverall,
        isTargeted: true,
        hasSubmitted,
        submission: latestRoundOverall.submissions[0] || null,
        reason: hasSubmitted ? 'LATEST_ACTIVE_ROUND_ALREADY_SUBMITTED' : 'LATEST_ACTIVE_ROUND_READY_FOR_TRAINING',
      };
    }

    // Client is NOT targeted for latest active round.
    // Check if there is another contemporary active round (same or adjacent roundNumber) that DOES target this client
    const alternativeActive = await prisma.trainingRound.findFirst({
      where: {
        status: 'ACTIVE',
        roundNumber: { gte: Math.max(1, latestRoundOverall.roundNumber - 1) },
        id: { not: latestRoundOverall.id },
      },
      orderBy: { roundNumber: 'desc' },
      include: {
        baseModel: true,
        submissions: {
          where: { clientId },
          orderBy: { submittedAt: 'desc' },
          take: 1,
          include: { arcScores: { orderBy: { calculatedAt: 'desc' }, take: 1 } },
        },
      },
    });

    if (alternativeActive && isClientTargeted(alternativeActive, clientId)) {
      const altSubmitted = alternativeActive.submissions.length > 0;
      return {
        round: alternativeActive,
        isTargeted: true,
        hasSubmitted: altSubmitted,
        submission: alternativeActive.submissions[0] || null,
        reason: altSubmitted ? 'LATEST_ACTIVE_ROUND_ALREADY_SUBMITTED' : 'LATEST_ACTIVE_ROUND_READY_FOR_TRAINING',
      };
    }

    return {
      round: latestRoundOverall,
      isTargeted: false,
      hasSubmitted: false,
      submission: null,
      reason: 'CLIENT_NOT_TARGETED',
    };
  }

  // If latest round overall is NOT ACTIVE (e.g. COMPLETED or ARCHIVED):
  // All rounds prior to it are strictly historical. Stale ACTIVE rounds with roundNumber < latest are ignored.
  return {
    round: latestRoundOverall,
    isTargeted: isClientTargeted(latestRoundOverall, clientId),
    hasSubmitted: latestRoundOverall.submissions.length > 0,
    submission: latestRoundOverall.submissions[0] || null,
    reason: latestRoundOverall.status === 'COMPLETED' ? 'LATEST_ROUND_COMPLETED' : 'LATEST_ROUND_INACTIVE',
  };
}

/**
 * Lists all active rounds with client-specific eligibility and submission status.
 * Filters out stale rounds older than recent active generation.
 * @param {string} clientId
 * @returns {Promise<Array<object>>}
 */
async function getActiveRoundsSummaryForClient(clientId) {
  const latestRound = await prisma.trainingRound.findFirst({
    orderBy: { roundNumber: 'desc' },
    select: { roundNumber: true },
  });

  if (!latestRound) return [];

  // Ignore stale active rounds that are more than 2 rounds behind the latest round
  const minRoundNumber = Math.max(1, latestRound.roundNumber - 2);

  const activeRounds = await prisma.trainingRound.findMany({
    where: {
      status: 'ACTIVE',
      roundNumber: { gte: minRoundNumber },
    },
    orderBy: { roundNumber: 'desc' },
    include: {
      submissions: {
        where: { clientId },
        select: { id: true, status: true, submittedAt: true, localAccuracy: true, checksumSha256: true },
      },
    },
  });

  return activeRounds.map((r) => {
    const targeted = isClientTargeted(r, clientId);
    const sub = r.submissions[0] || null;
    return {
      roundId: r.id,
      roundNumber: r.roundNumber,
      name: r.name || `Round #${r.roundNumber}`,
      status: r.status,
      isTargeted: targeted,
      hasSubmitted: Boolean(sub),
      submission: sub,
      canTrain: targeted && !sub && r.status === 'ACTIVE',
    };
  });
}

module.exports = {
  isClientTargeted,
  resolveEligibleRoundForClient,
  getActiveRoundsSummaryForClient,
};
