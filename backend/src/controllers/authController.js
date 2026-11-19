// Auth Controller - Registration, Login, and Profile endpoints
// Uses bcryptjs for password hashing, JWT for session tokens

const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const { validationResult } = require('express-validator');
const prisma = require('../config/database');
const config = require('../config');

const BCRYPT_ROUNDS = 12;

/**
 * Generate JWT token for a user.
 */
function signToken(user) {
  return jwt.sign(
    { sub: user.id, email: user.email, role: user.role },
    config.jwt.secret,
    { expiresIn: config.jwt.expiresIn }
  );
}

/**
 * POST /api/auth/register
 * Creates a new user account. If role=CLIENT, also creates a ClientProfile.
 */
async function register(req, res, next) {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ error: 'Validation failed', errors: errors.array() });
    }

    const { name, email, password, role = 'CLIENT' } = req.body;

    // Check for duplicate email
    const existing = await prisma.user.findUnique({ where: { email } });
    if (existing) {
      return res.status(409).json({ error: 'Conflict', message: 'Email already registered' });
    }

    const passwordHash = await bcrypt.hash(password, BCRYPT_ROUNDS);

    // Create user in a transaction to also create client profile if needed
    const user = await prisma.$transaction(async (tx) => {
      const newUser = await tx.user.create({
        data: { name, email, passwordHash, role },
      });

      if (role === 'CLIENT') {
        // Generate a sequential client ID like "client-001"
        const clientCount = await tx.clientProfile.count();
        const clientId = `client-${String(clientCount + 1).padStart(3, '0')}`;
        await tx.clientProfile.create({
          data: {
            userId: newUser.id,
            clientId,
            name,
          },
        });
      }

      return newUser;
    });

    const token = signToken(user);

    // Fetch with profile
    const fullUser = await prisma.user.findUnique({
      where: { id: user.id },
      include: { clientProfile: true },
      omit: { passwordHash: true },
    });

    res.status(201).json({ token, user: fullUser });
  } catch (err) {
    next(err);
  }
}

/**
 * POST /api/auth/login
 * Authenticates user and returns a JWT.
 */
async function login(req, res, next) {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ error: 'Validation failed', errors: errors.array() });
    }

    const { email, password } = req.body;

    const user = await prisma.user.findUnique({
      where: { email },
      include: { clientProfile: true },
    });

    if (!user) {
      return res.status(401).json({ error: 'Unauthorized', message: 'Invalid credentials' });
    }

    const passwordMatch = await bcrypt.compare(password, user.passwordHash);
    if (!passwordMatch) {
      return res.status(401).json({ error: 'Unauthorized', message: 'Invalid credentials' });
    }

    // Update last active timestamp for clients
    if (user.clientProfile) {
      await prisma.clientProfile.update({
        where: { id: user.clientProfile.id },
        data: { lastActiveAt: new Date() },
      });
    }

    const token = signToken(user);
    const { passwordHash, ...safeUser } = user;

    res.json({ token, user: safeUser });
  } catch (err) {
    next(err);
  }
}

/**
 * GET /api/auth/me
 * Returns the authenticated user's profile from req.user (set by verifyToken).
 */
async function getMe(req, res, next) {
  try {
    const { passwordHash, ...safeUser } = req.user;
    res.json({ user: safeUser });
  } catch (err) {
    next(err);
  }
}

module.exports = { register, login, getMe };
