/**
 * Client Dataset Pre-Training Audit Service
 *
 * Verifies class coverage across participating client datasets prior to training.
 * Detects missing classes per-client and across the entire participating client union.
 * Warns administrators if critical attack classes are missing from all clients,
 * which causes catastrophic forgetting and decision boundary collapse.
 *
 * SAFETY INVARIANT: Strictly quarantined from P:\project\test.csv (zero access).
 */

const fs = require('fs');
const path = require('path');
const config = require('../config');

/**
 * Loads the authoritative 34 class names from model configuration.
 */
function getAuthoritativeClasses() {
  try {
    const configPath = path.join(config.storage.root, 'global-models', 'initial-v1', 'model_config.json');
    if (fs.existsSync(configPath)) {
      const cfg = JSON.parse(fs.readFileSync(configPath, 'utf8'));
      if (Array.isArray(cfg.classes)) return cfg.classes;
    }
  } catch (err) {
    // Fallback to artifacts
  }

  try {
    const artPath = path.resolve(__dirname, '../../../ml-service/artifacts/model_config.json');
    if (fs.existsSync(artPath)) {
      const cfg = JSON.parse(fs.readFileSync(artPath, 'utf8'));
      if (Array.isArray(cfg.classes)) return cfg.classes;
    }
  } catch (err) {
    // Fallback to static list
  }

  return [
    'Backdoor_Malware', 'BenignTraffic', 'BrowserHijacking', 'CommandInjection',
    'DDoS-ACK_Fragmentation', 'DDoS-HTTP_Flood', 'DDoS-ICMP_Flood', 'DDoS-ICMP_Fragmentation',
    'DDoS-PSHACK_Flood', 'DDoS-RSTFINFlood', 'DDoS-SYN_Flood', 'DDoS-SlowLoris',
    'DDoS-SynonymousIP_Flood', 'DDoS-TCP_Flood', 'DDoS-UDP_Flood', 'DDoS-UDP_Fragmentation',
    'DNS_Spoofing', 'DoS-HTTP_Flood', 'DoS-SYN_Flood', 'DoS-TCP_Flood',
    'DoS-UDP_Flood', 'DictionaryBruteForce', 'MITM-ArpSpoofing', 'Mirai-greeth_flood',
    'Mirai-greip_flood', 'Mirai-udpplain', 'Recon-HostDiscovery', 'Recon-OSScan',
    'Recon-PingSweep', 'Recon-PortScan', 'SqlInjection', 'Uploading_Attack',
    'VulnerabilityScan', 'XSS',
  ];
}

/**
 * Resolves the assigned private dataset path for a client ID.
 */
function resolveClientDatasetPath(clientId) {
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
  return { filename, fullPath };
}

/**
 * Audit class distribution across participating client datasets.
 *
 * @param {string[]} clientIds Array of client identifiers (e.g. ['client-001', 'client-002'])
 * @returns {object} Detailed audit report with present, missing, and union deficit.
 */
function auditClientDatasets(clientIds = []) {
  const allExpectedClasses = getAuthoritativeClasses();
  const report = {
    evaluatedAt: new Date().toISOString(),
    totalExpectedClasses: allExpectedClasses.length,
    allExpectedClasses,
    clients: {},
    unionPresentClasses: [],
    missingAcrossAllClients: [],
    warnings: [],
    hasCriticalDeficit: false,
  };

  const unionSet = new Set();
  const targetIds = clientIds.length > 0 ? clientIds : ['client-001', 'client-002'];

  for (const cid of targetIds) {
    const { filename, fullPath } = resolveClientDatasetPath(cid);

    if (!fs.existsSync(fullPath)) {
      report.clients[cid] = {
        datasetFile: filename,
        error: `Dataset file ${filename} not found on disk.`,
        sampleCount: 0,
        uniqueClassCount: 0,
        presentClasses: [],
        missingClasses: allExpectedClasses,
        classCounts: {},
      };
      report.warnings.push({
        level: 'ERROR',
        clientId: cid,
        message: `Dataset ${filename} missing for ${cid}.`,
      });
      continue;
    }

    // Read and parse label column from CSV safely
    const content = fs.readFileSync(fullPath, 'utf8');
    const lines = content.trim().split(/\r?\n/);
    if (lines.length < 2) {
      report.clients[cid] = {
        datasetFile: filename,
        error: `Dataset file ${filename} is empty or has no data rows.`,
        sampleCount: 0,
        uniqueClassCount: 0,
        presentClasses: [],
        missingClasses: allExpectedClasses,
        classCounts: {},
      };
      continue;
    }

    const header = lines[0].split(',').map((s) => s.trim().replace(/^["']|["']$/g, ''));
    const labelIdx = header.indexOf('label');

    if (labelIdx === -1) {
      report.clients[cid] = {
        datasetFile: filename,
        error: `No 'label' column found in ${filename}.`,
        sampleCount: lines.length - 1,
        uniqueClassCount: 0,
        presentClasses: [],
        missingClasses: allExpectedClasses,
        classCounts: {},
      };
      continue;
    }

    const classCounts = {};
    for (let i = 1; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line) continue;
      const cols = line.split(',');
      const lbl = cols[labelIdx] ? cols[labelIdx].trim().replace(/^["']|["']$/g, '') : null;
      if (lbl) {
        classCounts[lbl] = (classCounts[lbl] || 0) + 1;
        unionSet.add(lbl);
      }
    }

    const present = Object.keys(classCounts).sort();
    const missing = allExpectedClasses.filter((c) => !classCounts[c]).sort();

    report.clients[cid] = {
      datasetFile: filename,
      sampleCount: lines.length - 1,
      uniqueClassCount: present.length,
      presentClasses: present,
      missingClasses: missing,
      classCounts,
    };
  }

  report.unionPresentClasses = Array.from(unionSet).sort();
  report.missingAcrossAllClients = allExpectedClasses.filter((c) => !unionSet.has(c)).sort();

  if (report.missingAcrossAllClients.length > 0) {
    report.hasCriticalDeficit = true;
    report.warnings.push({
      level: 'CRITICAL',
      message: `Severe class deficit: ${report.missingAcrossAllClients.length} of ${allExpectedClasses.length} attack classes are absent from EVERY participating client dataset.`,
      missingClasses: report.missingAcrossAllClients,
      recommendation:
        'Local training without these classes will cause gradient drift and catastrophic forgetting. Use conservative learning rates (lr <= 0.0001), enable proximal regularization (mu > 0), or include clients covering the missing categories.',
    });
  }

  return report;
}

module.exports = {
  auditClientDatasets,
  getAuthoritativeClasses,
};
