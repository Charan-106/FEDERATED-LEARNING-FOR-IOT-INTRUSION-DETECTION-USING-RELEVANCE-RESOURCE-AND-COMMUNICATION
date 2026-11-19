"""
ARC Scoring Engine - Version 1.0

Implements the Adaptive Resource-aware Contribution (ARC) scoring formula:

    ARC_i = lambda_R * R_i + lambda_S * S_i + lambda_C * C_i

Where:
    R_i = Relevance Score    = 0.60 * Q_i + 0.40 * D_i
    S_i = Resource Score     = 0.40 * M_i + 0.40 * P_i + 0.20 * A_i
    C_i = Communication Score = 0.50 * B_i + 0.30 * L_i + 0.20 * T_i

All component scores are normalized to [0, 100].

Default weights: lambda_R=0.40, lambda_S=0.30, lambda_C=0.30
"""

import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ARCScoringEngine:
    """
    Computes ARC scores for federated learning clients.
    
    The scoring engine is initialized with weights from the ARCConfig table
    so that administrators can tune the formula without code changes.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Dict from ARCConfig DB record containing all lambda/weight values.
                    If not provided, uses standard defaults.
        """
        # Top-level lambda weights (must sum to 1.0)
        self.lambda_relevance = config.get("lambdaRelevance", config.get("lambda_relevance", 0.40))
        self.lambda_resource = config.get("lambdaResource", config.get("lambda_resource", 0.30))
        self.lambda_communication = config.get("lambdaCommunication", config.get("lambda_communication", 0.30))

        # Relevance sub-weights: R_i = w_Q*Q_i + w_D*D_i
        self.w_quality = config.get("weightQuality", config.get("weight_quality", 0.60))
        self.w_diversity = config.get("weightDiversity", config.get("weight_diversity", 0.40))

        # Resource sub-weights: S_i = w_M*M_i + w_P*P_i + w_A*A_i
        self.w_memory = config.get("weightMemory", config.get("weight_memory", 0.40))
        self.w_processing = config.get("weightProcessing", config.get("weight_processing", 0.40))
        self.w_capacity = config.get("weightCapacity", config.get("weight_capacity", 0.20))

        # Communication sub-weights: C_i = w_B*B_i + w_L*L_i + w_T*T_i
        self.w_bandwidth = config.get("weightBandwidth", config.get("weight_bandwidth", 0.50))
        self.w_latency = config.get("weightLatency", config.get("weight_latency", 0.30))
        self.w_reliability = config.get("weightReliability", config.get("weight_reliability", 0.20))

    def compute_relevance(self, quality_metrics: dict, diversity_metrics: dict) -> dict:
        """
        Compute the Relevance score R_i = w_Q*Q_i + w_D*D_i

        Quality metrics (data integrity):
            - quality_score: optional explicit score in [0, 100] (client override/manual input)
            - missing_ratio: fraction of NaN/missing values [0,1] → lower is better
            - inf_ratio: fraction of Inf values [0,1] → lower is better  
            - preprocessing_success: bool → 1.0 if scaler worked correctly

        Diversity metrics (class distribution):
            - diversity_score: optional explicit score in [0, 100] (client override/manual input)
            - class_count: number of unique classes in client data [0, 34]
            - entropy: Shannon entropy of class distribution [0, log2(34)]
            - minority_class_coverage: fraction of rare classes present [0,1]
        """
        # --- Quality Score Q_i ---
        # Multi-factor evaluation: Completeness (missing/inf), Feature Noise, and Reference Model Label Agreement
        q_components = []
        q_weights = []

        raw_missing = quality_metrics.get("missing_ratio")
        raw_inf = quality_metrics.get("inf_ratio")
        raw_noise_score = quality_metrics.get("feature_noise_score")
        raw_label_score = quality_metrics.get("label_agreement_score", quality_metrics.get("label_consistency_score"))
        preprocessing_ok = float(quality_metrics.get("preprocessing_success", 1.0) or 1.0)

        # 1. Data Completeness (missing cells and infinite values)
        if raw_missing is not None or raw_inf is not None:
            missing_ratio = float(raw_missing or 0.0)
            inf_ratio = float(raw_inf or 0.0)
            completeness = max(0.0, 1.0 - missing_ratio - inf_ratio) * preprocessing_ok * 100.0
            q_components.append(completeness * 0.40)
            q_weights.append(0.40)
        else:
            completeness = None

        # 2. Feature Noise Outlier Score (z-score domain violation)
        if raw_noise_score is not None:
            q_noise = float(raw_noise_score)
            q_components.append(q_noise * 0.30)
            q_weights.append(0.30)
        else:
            q_noise = None

        # 3. Reference Model Label Agreement Score (out-of-fold reference inference heuristic)
        if raw_label_score is not None:
            q_label = float(raw_label_score)
            q_components.append(q_label * 0.30)
            q_weights.append(0.30)
        else:
            q_label = None

        if q_weights and sum(q_weights) > 0:
            Q_i = sum(q_components) / sum(q_weights)
            quality_provenance = "MEASURED_MULTIFACTOR_INTEGRITY"
        elif quality_metrics.get("quality_score") is not None:
            Q_i = float(quality_metrics["quality_score"])
            completeness = Q_i
            quality_provenance = "CLIENT_CONFIGURED_OVERRIDE"
        else:
            Q_i = None
            completeness = None
            quality_provenance = "MISSING_UNAVAILABLE"

        # --- Diversity Score D_i ---
        # Prioritize genuinely measured class count and Shannon entropy
        if diversity_metrics.get("class_count") is not None or diversity_metrics.get("entropy") is not None:
            class_count = float(diversity_metrics.get("class_count", 1) or 1)
            max_classes = 34.0  # CICIoT2023 has 34 classes
            class_coverage = min(class_count / max_classes, 1.0) * 100.0

            entropy = float(diversity_metrics.get("entropy", 0.0) or 0.0)
            max_entropy = math.log2(max_classes)
            entropy_score = min(entropy / max_entropy, 1.0) * 100.0 if max_entropy > 0 else 0.0

            minority_coverage = float(diversity_metrics.get("minority_class_coverage", 0.0) or 0.0) * 100.0
            D_i = (class_coverage + entropy_score + minority_coverage) / 3.0
            diversity_provenance = "SHANNON_ENTROPY_DISTRIBUTION_PROXY"
        elif diversity_metrics.get("diversity_score") is not None:
            D_i = float(diversity_metrics["diversity_score"])
            class_coverage = D_i
            entropy_score = D_i
            minority_coverage = D_i
            diversity_provenance = "CLIENT_CONFIGURED_OVERRIDE"
        else:
            class_coverage = None
            entropy_score = None
            minority_coverage = None
            D_i = None
            diversity_provenance = "MISSING_UNAVAILABLE"

        # --- Composite Relevance ---
        r_components = []
        r_weights = []
        if Q_i is not None:
            r_components.append(self.w_quality * Q_i)
            r_weights.append(self.w_quality)
        if D_i is not None:
            r_components.append(self.w_diversity * D_i)
            r_weights.append(self.w_diversity)

        if r_weights and sum(r_weights) > 0:
            R_i = sum(r_components) / sum(r_weights)
        else:
            R_i = None

        return {
            "relevance_score": round(R_i, 4) if R_i is not None else None,
            "quality_score": round(Q_i, 4) if Q_i is not None else None,
            "diversity_score": round(D_i, 4) if D_i is not None else None,
            "quality_components": {
                "completeness": round(completeness, 4) if completeness is not None else None,
                "feature_noise": round(q_noise, 4) if q_noise is not None else None,
                "label_agreement": round(q_label, 4) if q_label is not None else None,
                "label_consistency": round(q_label, 4) if q_label is not None else None,
                "preprocessing_ok": preprocessing_ok,
            },
            "diversity_components": {
                "class_coverage": round(class_coverage, 4) if class_coverage is not None else None,
                "entropy_score": round(entropy_score, 4) if entropy_score is not None else None,
                "minority_coverage": round(minority_coverage, 4) if minority_coverage is not None else None,
            },
            "metric_provenance": {
                "quality": quality_provenance,
                "diversity": diversity_provenance,
                "resource": "SIMULATION_HOST_PSUTIL",
                "communication": "SIMULATION_TELEMETRY_ESTIMATE",
                "interpretability_note": "Multi-factor quality measures data completeness, z-score noise, and reference model label agreement. Disagreement with the reference model is strictly a heuristic alignment measure and does not constitute proof of label poisoning."
            },
        }

    def compute_resource(self, resource_metrics: dict) -> dict:
        """
        Compute the Resource score S_i = w_M*M_i + w_P*P_i + w_A*A_i

        Resource metrics:
            - available_ram_gb: free RAM on the client device
            - total_ram_gb: total RAM (used for normalization)
            - cpu_idle_pct: idle CPU percentage [0, 100] → higher = more capacity
            - throughput_samples_per_sec: local training throughput

        If any measurement cannot be captured, it is marked unavailable (None)
        instead of inventing a value, and composite score scales over available measurements.
        """
        raw_avail_ram = resource_metrics.get("available_ram_gb")
        raw_total_ram = resource_metrics.get("total_ram_gb")
        raw_cpu_idle = resource_metrics.get("cpu_idle_pct")
        raw_throughput = resource_metrics.get("throughput_samples_per_sec")

        valid_components = []
        valid_weights = []

        # M_i: Memory availability score
        M_i = None
        if raw_avail_ram is not None and raw_total_ram is not None and float(raw_total_ram) > 0:
            M_i = round(min(float(raw_avail_ram) / float(raw_total_ram), 1.0) * 100.0, 4)
            valid_components.append(M_i * self.w_memory)
            valid_weights.append(self.w_memory)

        # P_i: Processing score (idle CPU %)
        P_i = None
        if raw_cpu_idle is not None:
            P_i = round(max(0.0, min(float(raw_cpu_idle), 100.0)), 4)
            valid_components.append(P_i * self.w_processing)
            valid_weights.append(self.w_processing)

        # A_i: Throughput score
        A_i = None
        if raw_throughput is not None:
            THROUGHPUT_REFERENCE = 1000.0
            A_i = round(min(float(raw_throughput) / THROUGHPUT_REFERENCE, 1.0) * 100.0, 4)
            valid_components.append(A_i * self.w_capacity)
            valid_weights.append(self.w_capacity)

        if valid_weights and sum(valid_weights) > 0:
            S_i = round(sum(valid_components) / sum(valid_weights), 4)
        else:
            S_i = None

        return {
            "resource_score": S_i,
            "memory_score": M_i,
            "processing_score": P_i,
            "capacity_score": A_i,
        }

    def compute_communication(self, comm_metrics: dict) -> dict:
        """
        Compute the Communication score C_i = w_B*B_i + w_L*L_i + w_T*T_i

        Communication metrics:
            - bandwidth_mbps: uplink bandwidth [Mbps] → higher is better
            - latency_ms: round-trip latency [ms] → lower is better
            - reliability_pct: packet delivery success rate [0, 100]

        If any measurement cannot be captured, it is marked unavailable (None)
        instead of inventing a value, and composite score scales over available measurements.
        """
        raw_bw = comm_metrics.get("bandwidth_mbps")
        raw_lat = comm_metrics.get("latency_ms")
        raw_rel = comm_metrics.get("reliability_pct")

        valid_components = []
        valid_weights = []

        # B_i: Bandwidth score
        B_i = None
        if raw_bw is not None:
            BANDWIDTH_REFERENCE = 100.0
            B_i = round(min(float(raw_bw) / BANDWIDTH_REFERENCE, 1.0) * 100.0, 4)
            valid_components.append(B_i * self.w_bandwidth)
            valid_weights.append(self.w_bandwidth)

        # L_i: Latency score
        L_i = None
        if raw_lat is not None:
            MAX_LATENCY_MS = 500.0
            L_i = round(max(0.0, (1.0 - float(raw_lat) / MAX_LATENCY_MS)) * 100.0, 4)
            valid_components.append(L_i * self.w_latency)
            valid_weights.append(self.w_latency)

        # T_i: Reliability score
        T_i = None
        if raw_rel is not None:
            T_i = round(max(0.0, min(float(raw_rel), 100.0)), 4)
            valid_components.append(T_i * self.w_reliability)
            valid_weights.append(self.w_reliability)

        if valid_weights and sum(valid_weights) > 0:
            C_i = round(sum(valid_components) / sum(valid_weights), 4)
        else:
            C_i = None

        return {
            "communication_score": C_i,
            "bandwidth_score": B_i,
            "latency_score": L_i,
            "reliability_score": T_i,
        }

    def compute_overall(self, R: Optional[float], S: Optional[float], C: Optional[float]) -> float:
        """
        Compute the overall ARC score:
            ARC_i = lambda_R * R_i + lambda_S * S_i + lambda_C * C_i

        Normalizes dynamically across available components.
        Returns the score in [0, 100].
        """
        valid_components = []
        valid_weights = []

        if R is not None:
            valid_components.append(self.lambda_relevance * float(R))
            valid_weights.append(self.lambda_relevance)
        if S is not None:
            valid_components.append(self.lambda_resource * float(S))
            valid_weights.append(self.lambda_resource)
        if C is not None:
            valid_components.append(self.lambda_communication * float(C))
            valid_weights.append(self.lambda_communication)

        if valid_weights and sum(valid_weights) > 0:
            arc = sum(valid_components) / sum(valid_weights)
            return round(min(max(arc, 0.0), 100.0), 4)
        return 0.0

    def score(
        self,
        quality_metrics: dict,
        diversity_metrics: dict,
        resource_metrics: dict,
        comm_metrics: dict,
    ) -> dict:
        """
        Full ARC scoring pipeline. Computes all components and returns
        a complete breakdown for storage in the ARCScore table.
        """
        relevance_result = self.compute_relevance(quality_metrics, diversity_metrics)
        resource_result = self.compute_resource(resource_metrics)
        comm_result = self.compute_communication(comm_metrics)

        R_i = relevance_result["relevance_score"]
        S_i = resource_result["resource_score"]
        C_i = comm_result["communication_score"]

        arc_i = self.compute_overall(R_i, S_i, C_i)

        return {
            "overall_arc_score": arc_i,
            "relevance_score": R_i,
            "resource_score": S_i,
            "communication_score": C_i,
            # Individual sub-scores
            "quality_score": relevance_result["quality_score"],
            "diversity_score": relevance_result["diversity_score"],
            "memory_score": resource_result["memory_score"],
            "processing_score": resource_result["processing_score"],
            "capacity_score": resource_result["capacity_score"],
            "bandwidth_score": comm_result["bandwidth_score"],
            "latency_score": comm_result["latency_score"],
            "reliability_score": comm_result["reliability_score"],
            # Weights snapshot for auditability
            "weights_used": {
                "lambda_relevance": self.lambda_relevance,
                "lambda_resource": self.lambda_resource,
                "lambda_communication": self.lambda_communication,
            },
            # Detailed breakdowns
            "quality_components": relevance_result.get("quality_components"),
            "diversity_components": relevance_result.get("diversity_components"),
        }


# Default scoring engine using standard weights
default_engine = ARCScoringEngine({})
