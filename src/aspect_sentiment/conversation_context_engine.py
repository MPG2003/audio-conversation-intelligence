def analyze_context(transcript: str, raw_features: list) -> dict:
    """
    Analyzes the depth and structure of the conversation.
    Longer engaged conversations should increase confidence.
    """
    words = transcript.split()
    length = len(words)
    
    # Estimate turns (very rough approximation by looking at punctuation / sentence breaks)
    # A real system would use diarization
    turns = transcript.count("?") + transcript.count(".") + transcript.count("!")
    
    technical_features_count = len([f for f in raw_features if f.get("label") == "FEATURE"])
    
    score = 0.0
    
    # 1. Transcript length baseline
    if length > 200:
        score += 0.4
    elif length > 100:
        score += 0.2
    elif length > 40:
        score += 0.1
        
    # 2. Conversational turns (engagement)
    if turns > 10:
        score += 0.3
    elif turns > 5:
        score += 0.15
        
    # 3. Technical depth
    if technical_features_count > 3:
        score += 0.3
    elif technical_features_count > 1:
        score += 0.15
        
    # Normalize to 0-1
    engagement_score = min(1.0, score)
    
    return {
        "engagement_score": engagement_score,
        "word_count": length,
        "estimated_turns": turns,
        "technical_depth": technical_features_count
    }
