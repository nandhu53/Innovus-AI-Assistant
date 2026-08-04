import streamlit as st
from supabase import create_client, Client
import os

# Your existing Supabase setup
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

with st.sidebar:
    st.title("🔐 User Authentication")
    
    # Standard Email/Password form here (optional to keep)
    # ...
    
    st.divider()
    
    # The New Google SSO Button
    st.markdown("### Or sign in with:")
    
    if st.button("🌐 Continue with Google"):
        try:
            # Tell Supabase to trigger the Google OAuth flow
            response = supabase.auth.sign_in_with_oauth({
                "provider": "google",
                "options": {
                    # Redirects them back to your Streamlit app after login
                    "redirect_to": "https://innovus-ai-assistant-yourlink.streamlit.app" 
                }
            })
            
            # Streamlit workaround to redirect the user to the Google login page
            st.markdown(f'<meta http-equiv="refresh" content="0;url={response.url}">', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Google login failed: {e}")
