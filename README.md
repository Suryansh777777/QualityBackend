https://drive.google.com/file/d/1q-4tBJWAVM5-Od7G4ITyiQO-zJixbUXe/view
sudo nano /etc/systemd/system/fruit-analysis.service
[Unit]
Description=Fruit Analysis Server
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/your-project-directory
ExecStart=/usr/bin/python3 /home/pi/your-project-directory/app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
sudo chmod 644 /etc/systemd/system/fruit-analysis.service

sudo systemctl enable fruit-analysis.service
sudo systemctl start fruit-analysis.service

sudo systemctl status fruit-analysis.service

ExecStart=/home/pi/your-project-directory/venv/bin/python /home/pi/your-project-directory/app.py

[Service]
Environment=FLASK_ENV=production
Environment=PORT=5000
