// Model routes: /api/models/
const express = require('express');
const { verifyToken, requireRole } = require('../middleware/authMiddleware');
const {
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
} = require('../controllers/modelController');

const router = express.Router();

// Authoritative model references & current model available to authenticated users
router.get('/current', verifyToken, getCurrentModel);
router.get('/references', verifyToken, getModelReferences);
router.get('/compare', verifyToken, requireRole('ADMIN'), compareModelsController);
router.get('/versions', verifyToken, requireRole('ADMIN'), getVersions);
router.get('/details/:modelId', verifyToken, requireRole('ADMIN'), getModelDetails);
router.get('/:modelId/download', verifyToken, downloadVersion);
router.post('/publish-candidate', verifyToken, requireRole('ADMIN'), publishCandidate);
router.post('/publish-candidate-and-start-round', verifyToken, requireRole('ADMIN'), publishCandidateAndStartRound);
router.post('/reject-candidate', verifyToken, requireRole('ADMIN'), rejectCandidate);
router.post('/:modelId/publish', verifyToken, requireRole('ADMIN'), publishModel);
router.post('/:modelId/rollback', verifyToken, requireRole('ADMIN'), rollbackModel);

module.exports = router;
