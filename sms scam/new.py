import re
import pandas as pd
import numpy as np
import whois
import dns.resolver
import ssl
import socket
import urllib.parse
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib

# --- ML Model Functions ---
datapath = r"C:/Users/anju.ms/Learning/sms scam/spam.csv"  

import re

def extract_features(text):
    """Extract additional features from the SMS text."""
    features = {}
    
    # Check for phone numbers
    phone_pattern = r'\b(?:\+?(\d{1,3}))?[-. (]*(\d{3})[-. )]*(\d{3})[-. ]*(\d{4})\b'
    phone_numbers = re.findall(phone_pattern, text)
    features['has_phone_number'] = len(phone_numbers) > 0
    
    # Check for scam-related keywords
    scam_keywords = [
        'call now', 'reply to claim', 'win', 'prize', 'urgent', 'limited time', 
        'congratulations', 'free', 'gift', 'account', 'verify', 'click', 'offer'
    ]
    features['scam_keyword_count'] = sum(keyword in text.lower() for keyword in scam_keywords)
    
    # Check for urgency-related phrases
    urgency_phrases = ['act now', 'limited time', 'urgent', 'immediately']
    features['urgency_score'] = sum(phrase in text.lower() for phrase in urgency_phrases)
    
    # Check for poor grammar (e.g., excessive capitalization)
    features['excessive_caps'] = sum(1 for char in text if char.isupper()) / len(text) > 0.5
    
    return features

def train_spam_model(data_path):
    """Train a spam detection model using SMS dataset."""
    print(f"Loading dataset from {data_path}...")
    
    # Load dataset with the correct encoding
    df = pd.read_csv(data_path, encoding='latin-1')
    
    # Ensure we have the right columns
    if 'v1' not in df.columns or 'v2' not in df.columns:
        raise ValueError("Dataset must contain 'v1' (label) and 'v2' (message) columns")
    
    # Rename columns for clarity
    df = df.rename(columns={'v1': 'label', 'v2': 'message'})
    
    # Extract additional features
    df['features'] = df['message'].apply(extract_features)
    
    # Convert features into separate columns
    df = pd.concat([df, pd.json_normalize(df['features'])], axis=1)
    df = df.drop(columns=['features'])
    
    print(f"Dataset loaded: {len(df)} messages ({df['label'].value_counts().to_dict()})")
    
    # Split data
    X = df[['message', 'has_phone_number', 'scam_keyword_count', 'urgency_score', 'excessive_caps']]
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("Training model...")
    
    # Create and fit the vectorizer for text data
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_text = vectorizer.fit_transform(X_train['message'])
    X_test_text = vectorizer.transform(X_test['message'])
    
    # Combine text features with additional features
    X_train_combined = np.hstack([X_train_text.toarray(), X_train[['has_phone_number', 'scam_keyword_count', 'urgency_score', 'excessive_caps']]])
    X_test_combined = np.hstack([X_test_text.toarray(), X_test[['has_phone_number', 'scam_keyword_count', 'urgency_score', 'excessive_caps']]])
    
    # Train the model
    model = MultinomialNB()
    model.fit(X_train_combined, y_train)
    
    # Test the model
    y_pred = model.predict(X_test_combined)
    accuracy = np.mean(y_pred == y_test)
    print(f"Model accuracy: {accuracy:.2f}")
    
    # Save the model
    model_path = 'sms_spam_model.pkl'
    joblib.dump({'model': model, 'vectorizer': vectorizer}, model_path)
    print(f"Model saved to {model_path}")
    
    return model, vectorizer

# def train_spam_model(data_path):
#     """Train a spam detection model using SMS dataset."""
#     print(f"Loading dataset from {data_path}...")
    
#     # Load dataset - adapt to your file format
#     df = pd.read_csv(data_path, encoding='latin-1')
    
#     # Ensure we have the right columns
#     if 'v1' not in df.columns or 'v2' not in df.columns:
#         raise ValueError("Dataset must contain 'v1' (label) and 'v2' (message) columns")
    
#     # Rename columns for clarity
#     df = df.rename(columns={'v1': 'label', 'v2': 'message'})
    
#     print(f"Dataset loaded: {len(df)} messages ({df['label'].value_counts().to_dict()})")
    
#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(
#         df['message'], df['label'], test_size=0.2, random_state=42
#     )
    
#     print("Training model...")
    
#     # Create and fit the vectorizer
#     vectorizer = TfidfVectorizer(max_features=5000)
#     X_train_vec = vectorizer.fit_transform(X_train)
#     X_test_vec = vectorizer.transform(X_test)
    
#     # Train the model
#     model = MultinomialNB()
#     model.fit(X_train_vec, y_train)
    
#     # Test the model
#     y_pred = model.predict(X_test_vec)
#     accuracy = np.mean(y_pred == y_test)
#     print(f"Model accuracy: {accuracy:.2f}")
    
#     # Save the model
#     model_path = 'sms_spam_model.pkl'
#     joblib.dump({'model': model, 'vectorizer': vectorizer}, model_path)
#     print(f"Model saved to {model_path}")
    
#     return model, vectorizer

def load_spam_model(model_path='sms_spam_model.pkl'):
    """Load a pre-trained spam detection model."""
    print(f"Loading model from {model_path}...")
    loaded = joblib.load(model_path)
    return loaded['model'], loaded['vectorizer']

def predict_spam(sms_text, model, vectorizer):
    """Predict if an SMS is spam or not."""
    # Vectorize the SMS
    sms_vec = vectorizer.transform([sms_text])
    
    # Predict
    prediction = model.predict(sms_vec)[0]
    probabilities = model.predict_proba(sms_vec)[0]
    
    # If 'spam' is at index 1
    if model.classes_[1] == 'spam':
        spam_probability = probabilities[1]
    else:
        spam_probability = probabilities[0]
    
    return {
        'is_spam': prediction == 'spam',
        'spam_probability': spam_probability,
        'prediction': prediction.upper(),
        'confidence': spam_probability if prediction == 'spam' else 1 - spam_probability
    }

# --- URL Analysis Functions ---

def extract_urls(text):
    """Extract URLs from text."""
    url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+|[a-zA-Z0-9][a-zA-Z0-9-]{1,61}[a-zA-Z0-9]\.[a-zA-Z]{2,}(?:/[^\s<>"]*)?'
    
    # Find all URLs
    urls = re.findall(url_pattern, text)
    
    # Ensure URLs have http/https prefix
    processed_urls = []
    for url in urls:
        if not url.startswith(('http://', 'https://')):
            if url.startswith('www.'):
                url = 'http://' + url
            else:
                url = 'http://' + url
        processed_urls.append(url)
        
    return processed_urls

def analyze_url(url, timeout=5):
    """Analyze a URL for signs of being malicious without visiting it."""
    print(f"Analyzing URL: {url}")
    result = {
        'url': url,
        'risk_factors': [],
        'risk_score': 0
    }
    
    # Parse URL
    parsed_url = urllib.parse.urlparse(url)
    domain = parsed_url.netloc
    if domain.startswith('www.'):
        domain = domain[4:]
    
    # Check domain length (very long domains are suspicious)
    if len(domain) > 40:
        result['risk_factors'].append('Unusually long domain name')
        result['risk_score'] += 1
    
    # Check for suspicious TLDs
    suspicious_tlds = ['.xyz', '.top', '.loan', '.work', '.click', '.gdn']
    if any(domain.endswith(tld) for tld in suspicious_tlds):
        result['risk_factors'].append('Suspicious TLD')
        result['risk_score'] += 1
    
    # Check for domain age using WHOIS
    try:
        print(f"Checking WHOIS for {domain}...")
        w = whois.whois(domain)
        
        # Check creation date
        if w.creation_date:
            if isinstance(w.creation_date, list):
                creation_date = w.creation_date[0]
            else:
                creation_date = w.creation_date
            
            # Calculate domain age in days
            domain_age = (datetime.now() - creation_date).days
            result['domain_age_days'] = domain_age
            
            # New domains (less than 30 days) are suspicious
            if domain_age < 30:
                result['risk_factors'].append(f'Very new domain (age: {domain_age} days)')
                result['risk_score'] += 2
            elif domain_age < 90:
                result['risk_factors'].append(f'Relatively new domain (age: {domain_age} days)')
                result['risk_score'] += 1
            
            print(f"Domain age: {domain_age} days")
        
        # Store registrar information
        if w.registrar:
            result['registrar'] = w.registrar
            
    except Exception as e:
        print(f"WHOIS error: {str(e)}")
        result['risk_factors'].append('Could not retrieve WHOIS information')
        result['risk_score'] += 1
    
    # Check SSL certificate
    if url.startswith('https://'):
        try:
            print(f"Checking SSL for {domain}...")
            # Get SSL certificate without visiting the site
            hostname = domain
            context = ssl.create_default_context()
            with socket.create_connection((hostname, 443), timeout=timeout) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
            
            # Check if certificate is issued to different domain
            if 'subjectAltName' in cert:
                alt_names = [x[1] for x in cert['subjectAltName'] if x[0] == 'DNS']
                if domain not in alt_names and not any(domain.endswith('.' + name) for name in alt_names if name.startswith('*.')):
                    result['risk_factors'].append('Certificate domain mismatch')
                    result['risk_score'] += 2
            
            print("SSL certificate verified")
        except Exception as e:
            print(f"SSL error: {str(e)}")
            result['risk_factors'].append('SSL certificate validation failed')
            result['risk_score'] += 1
    else:
        result['risk_factors'].append('Not using HTTPS')
        result['risk_score'] += 1
    
    # DNS lookups
    try:
        print(f"Performing DNS lookups for {domain}...")
        # Get A record
        a_records = dns.resolver.resolve(domain, 'A')
        result['ip_addresses'] = [str(r) for r in a_records]
        
        # Check if legit site would have MX records
        if any(major in domain for major in ['bank', 'paypal', 'amazon', 'ebay', 'netflix']):
            try:
                mx_records = dns.resolver.resolve(domain, 'MX')
                result['has_mx_records'] = True
            except dns.resolver.NoAnswer:
                result['has_mx_records'] = False
                result['risk_factors'].append('Major organization domain without MX records')
                result['risk_score'] += 1
    except Exception as e:
        print(f"DNS error: {str(e)}")
        result['risk_factors'].append('DNS resolution failed')
        result['risk_score'] += 1
    
    # Check for suspicious URL patterns
    suspicious_patterns = [
        'secure', 'account', 'login', 'signin', 'verify', 'banking', 
        'update', 'confirm', 'password', 'wallet', 'cryptocurrency'
    ]
    
    path = parsed_url.path.lower()
    if any(pattern in path for pattern in suspicious_patterns):
        result['risk_factors'].append('URL path contains suspicious keywords')
        result['risk_score'] += 1
    
    # Check for URL shorteners
    url_shorteners = [
        'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'is.gd', 
        'cli.gs', 'ow.ly', 'buff.ly', 'adf.ly', 'shorte.st'
    ]
    
    if any(shortener in domain for shortener in url_shorteners):
        result['risk_factors'].append('Uses URL shortener')
        result['risk_score'] += 2
    
    # Calculate final risk assessment
    if result['risk_score'] >= 5:
        result['risk_assessment'] = 'High Risk'
    elif result['risk_score'] >= 3:
        result['risk_assessment'] = 'Medium Risk'
    elif result['risk_score'] >= 1:
        result['risk_assessment'] = 'Low Risk'
    else:
        result['risk_assessment'] = 'Minimal Risk'
    
    return result

# --- Main Analysis Function ---

def analyze_sms(sms_text, model=None, vectorizer=None):
    """Analyze an SMS message for potential scams."""
    print(f"\nAnalyzing SMS: {sms_text}")
    results = {"sms": sms_text}
    
    # Step 1: Use ML model to predict if spam
    if model and vectorizer:
        spam_prediction = predict_spam(sms_text, model, vectorizer)
        results["spam_prediction"] = spam_prediction
        print(f"Spam prediction: {spam_prediction['prediction']} (confidence: {spam_prediction['confidence']:.2f})")
    
    # Step 2: Extract and analyze URLs
    urls = extract_urls(sms_text)
    results["urls_found"] = len(urls)
    results["urls"] = urls
    
    print(f"Found {len(urls)} URLs in the message")
    
    # Step 3: Analyze each URL
    url_analysis = []
    for url in urls:
        try:
            analysis = analyze_url(url)
            url_analysis.append(analysis)
        except Exception as e:
            print(f"Error analyzing URL {url}: {str(e)}")
            url_analysis.append({
                "url": url,
                "error": str(e),
                "risk_assessment": "Unable to analyze"
            })
    
    results["url_analysis"] = url_analysis
    
    # Step 4: Determine risk based on spam prediction and URL analysis
    risk_level = "Low"
    reasons = []
    
    # Check spam prediction
    if "spam_prediction" in results and results["spam_prediction"]["is_spam"]:
        if results["spam_prediction"]["confidence"] > 0.8:
            risk_level = "High"
            reasons.append("High confidence spam prediction")
        elif results["spam_prediction"]["confidence"] > 0.5:
            risk_level = "Medium"
            reasons.append("Medium confidence spam prediction")
        else:
            risk_level = "Low"
            reasons.append("Low confidence spam prediction")
    
    # Check URL analysis
    for analysis in url_analysis:
        if analysis.get("risk_assessment") == "High Risk":
            risk_level = "High"
            reasons.append(f"High risk URL: {analysis.get('url')}")
        elif analysis.get("risk_assessment") == "Medium Risk" and risk_level != "High":
            risk_level = "Medium"
            reasons.append(f"Medium risk URL: {analysis.get('url')}")
    
    results["risk_assessment"] = {
        "is_scam": risk_level in ["High", "Medium"],
        "confidence": 10 if risk_level == "High" else 5 if risk_level == "Medium" else 1,
        "reasons": reasons,
        "recommendation": "Avoid clicking any links" if risk_level in ["High", "Medium"] else "Message appears safe"
    }
    
    # Final output
    if results["risk_assessment"]["is_scam"]:
        print("\n⚠️  **Scam Detected**  ⚠️")
        print(f"Confidence: {results['risk_assessment']['confidence']}/10")
        print("Reasons:")
        for reason in results["risk_assessment"]["reasons"]:
            print(f" - {reason}")
        print(f"Recommendation: {results['risk_assessment']['recommendation']}")
    else:
        print("\n✅  **Safe Message**  ✅")
        print(f"Confidence: {results['risk_assessment']['confidence']}/10")
        print(f"Recommendation: {results['risk_assessment']['recommendation']}")
    
    return results


# --- Example Usage ---

if __name__ == "__main__":
    # Step 1: Train the model (if the model file doesn't exist)
    try:
        model, vectorizer = load_spam_model()
    except FileNotFoundError:
        print("Model file not found. Training the model...")
        model, vectorizer = train_spam_model(datapath)
    
    # Step 2: Analyze the SMS
    sms_text = input("Enter the SMS message: ")
    results = analyze_sms(sms_text, model, vectorizer)
    
    # Step 3: Output the result
    if results["risk_assessment"]["is_scam"]:
        print("\nFinal Result: This message is a SCAM.")
    else:
        print("\nFinal Result: This message is SAFE.")