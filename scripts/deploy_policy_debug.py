from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

print("Launching App...")
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print("App Launched Successfully")
simulation_app.close()
