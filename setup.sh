mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
port = $PORT\n\
[theme]\n\
primaryColor='#6eb52f'\n\
backgroundColor='#f0f0f5'
secondaryBackgroundColor='#e0e0ef'
textColor='#262730'
font='sans serif'
" > ~/.streamlit/config.toml