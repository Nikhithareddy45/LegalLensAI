from importlib import import_module

SemanticRiskDetector = import_module(
    "src.risk_detection.semantic"
).SemanticRiskDetector

RuleBasedRiskDetector = import_module(
    "src.risk_detection.rules"
).RuleBasedRiskDetector

RiskFusion = import_module(
    "src.risk_detection.fusion"
).RiskFusion

RiskMetrics = import_module(
    "src.risk_detection.metrics"
).RiskMetrics


def main():
    """Test risk detection functionality"""
    
    print("\n" + "="*60)
    print("CHECKPOINT 3: RISK DETECTION TESTING")
    print("="*60)
    
    # Test data
    test_texts = [
        "Liability is limited to the amount paid under this Agreement",
        "The Client shall pay all outstanding dues before termination",
        "Confidential information must not be disclosed to third parties",
        "This Agreement may be terminated by either party upon written notice"
    ]
    
    try:
        # Initialize models
        print("\n[STEP 1] Initializing Risk Detection Models...")
        semantic = SemanticRiskDetector()
        rules = RuleBasedRiskDetector()
        fusion = RiskFusion(semantic, rules)
        print("✓ Models initialized successfully")
        
        # Test on sample texts
        print("\n[STEP 2] Testing Risk Detection...")
        results = []
        
        for i, text in enumerate(test_texts):
            result = fusion.fuse(text)
            results.append(result)
            
            print(f"\n  Sample {i+1}: {text[:50]}...")
            print(f"    Overall Risk Score: {result['overall_semantic_risk']:.3f}")
            print(f"    Rules Triggered: {result['rule_triggered_count']}")
            print(f"    High Risk Categories: {result['high_risk_categories']}")
        
        # Compute statistics
        print("\n[STEP 3] Computing Statistics...")
        stats = RiskMetrics.compute_risk_statistics(results)
        
        print(f"✓ Average Overall Risk: {stats['avg_overall_risk']:.3f}")
        print(f"✓ Risk Distribution:")
        print(f"    Low: {stats['risk_distribution']['low']} samples")
        print(f"    Medium: {stats['risk_distribution']['medium']} samples")
        print(f"    High: {stats['risk_distribution']['high']} samples")
        
        print("\n" + "="*60)
        print("✓ CHECKPOINT 3 PASSED!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ CHECKPOINT 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
