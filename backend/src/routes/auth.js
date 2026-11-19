// Auth routes: /api/auth/
const express = require('express');
const { body } = require('express-validator');
const { register, login, getMe } = require('../controllers/authController');
const { verifyToken } = require('../middleware/authMiddleware');

const router = express.Router();

// Input validation rules
const registerRules = [
  body('name').trim().notEmpty().withMessage('Name is required'),
  body('email').isEmail().normalizeEmail().withMessage('Valid email required'),
  body('password').isLength({ min: 8 }).withMessage('Password must be at least 8 characters'),
  body('role').optional().isIn(['CLIENT', 'ADMIN']).withMessage('Role must be CLIENT or ADMIN'),
];

const loginRules = [
  body('email').isEmail().normalizeEmail().withMessage('Valid email required'),
  body('password').notEmpty().withMessage('Password is required'),
];

router.post('/register', registerRules, register);
router.post('/login', loginRules, login);
router.get('/me', verifyToken, getMe);

module.exports = router;
