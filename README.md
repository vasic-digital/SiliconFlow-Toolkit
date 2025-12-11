# SiliconFlow Toolkit

To configure OpenCode and Charm for SiliconFlow provider run `python3 install.py` script and follow the instructions.

Create a cron job to keep models updated:

```bash
# Add to crontab -e
0 3 * * * python3 ~/.config/update_siliconflow_models.py >> ~/.siliconflow_update.log 2>&1
```

