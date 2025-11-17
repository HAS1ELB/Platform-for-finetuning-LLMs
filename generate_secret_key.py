#!/usr/bin/env python3
"""
Generate secure SECRET_KEY for your .env file
Run: python generate_secret_key.py
"""
import secrets

def generate_secret_key(length=64):
    """Generate a cryptographically secure random key."""
    return secrets.token_urlsafe(length)

if __name__ == "__main__":
    key = generate_secret_key()
    print("\n" + "="*70)
    print("🔐 GENERATED SECRET_KEY")
    print("="*70)
    print(f"\n{key}\n")
    print("="*70)
    print("\n📝 Add this to your .env file:")
    print(f"\nSECRET_KEY={key}\n")
    print("⚠️  IMPORTANT: Never commit this key to GitHub!")
    print("="*70 + "\n")
    
    # Also generate a shorter one for quick testing
    short_key = secrets.token_hex(32)
    print("\n💡 Alternative (shorter format):")
    print(f"\nSECRET_KEY={short_key}\n")
