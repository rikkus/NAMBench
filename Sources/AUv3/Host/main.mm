// Front ends for the AUv3 overhead host.
//
// macOS: run the app's executable directly with arguments; it prints to stdout
// and exits. The app bundle exists only because an AUv3 extension has to live
// in one.
//
// iOS: launching the app runs the default sweep on a background thread, logs to
// stdout (visible with `devicectl device process launch --console`) and writes
// auv3-overhead.json into Documents, then exits.

#import <Foundation/Foundation.h>
#include <TargetConditionals.h>

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>

#include "NBAUHost.h"

namespace
{

std::vector<std::string> split(const std::string& s)
{
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ','))
    if (!item.empty())
      out.push_back(item);
  return out;
}

/// Command-line options, shared by both front ends: on macOS from argv, on iOS
/// from `devicectl device process launch ... -- <args>`. Returns false after
/// printing why, for anything it does not understand.
bool parse_args(const std::vector<std::string>& args, HostOptions& options)
{
  for (size_t i = 0; i < args.size(); i++)
  {
    const std::string& a = args[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= args.size())
      {
        std::fprintf(stderr, "%s needs a value\n", a.c_str());
        std::exit(1);
      }
      return args[++i];
    };
    if (a == "--model")
      options.modelPath = next();
    else if (a == "--input")
      options.inputPath = next();
    else if (a == "--json")
      options.jsonPath = next();
    else if (a == "--warmup")
      options.warmupSeconds = std::atof(next().c_str());
    else if (a == "--window")
      options.windowSeconds = std::atof(next().c_str());
    else if (a == "--rt-seconds")
      options.rtSeconds = std::atof(next().c_str());
    else if (a == "--blocks")
    {
      options.blockSizes.clear();
      for (const std::string& s : split(next()))
        options.blockSizes.push_back(std::atoi(s.c_str()));
    }
    else if (a == "--submodels")
    {
      options.submodels.clear();
      for (const std::string& s : split(next()))
        options.submodels.push_back(s == "nano" ? NbSubmodelNarrowest : NbSubmodelWidest);
    }
    else if (a == "--arms")
    {
      options.arms.clear();
      for (const std::string& s : split(next()))
      {
        int arm = 0;
        if (!nb_au_parse_arm(s, arm))
        {
          std::fprintf(stderr, "unknown arm %s (A, A1, B, C, D, F, Fi)\n", s.c_str());
          return false;
        }
        options.arms.push_back(arm);
      }
    }
    else if (a.rfind("-NS", 0) == 0 || a.rfind("-Apple", 0) == 0)
      i++; // launch-services noise
    else
    {
      std::fprintf(stderr,
                   "usage: NAMBenchAUHost --model file.nam [--input in.wav] [--json out.json]\n"
                   "       [--arms A,A1,B,C,D,F,Fi] [--submodels nano,standard] [--blocks 16,32,64,128,256]\n"
                   "       [--warmup s] [--window s] [--rt-seconds s]\n");
      return false;
    }
  }
  return true;
}

void run_on_thread_and_exit(HostOptions options)
{
  // A thread of its own, not the main thread: AU instantiation completes on a
  // queue the main thread may be needed to service. User-interactive QoS so the
  // timed loop is scheduled like nambench's.
  NSThread* thread = [[NSThread alloc] initWithBlock:^{
    const int status = nb_au_host_run(options);
    std::exit(status);
  }];
  thread.qualityOfService = NSQualityOfServiceUserInteractive;
  thread.stackSize = 8 << 20;
  [thread start];
}

} // namespace

#if TARGET_OS_OSX

int main(int argc, const char* argv[])
{
  @autoreleasepool
  {
    HostOptions options;
    NSString* resources = NSBundle.mainBundle.resourcePath;
    options.inputPath = std::string(resources.UTF8String) + "/input.wav";

    std::vector<std::string> args(argv + 1, argv + argc);
    if (!parse_args(args, options))
      return 1;
    if (options.modelPath.empty())
    {
      std::fprintf(stderr, "--model is required\n");
      return 1;
    }

    run_on_thread_and_exit(options);
    dispatch_main();
  }
}

#else // iOS

#import <UIKit/UIKit.h>

// iOS 27 requires the scene lifecycle; the window lives in the scene delegate.
@interface NBAUHostSceneDelegate : UIResponder <UIWindowSceneDelegate>
@property(strong, nonatomic) UIWindow* window;
@end

@implementation NBAUHostSceneDelegate

- (void)scene:(UIScene*)scene
    willConnectToSession:(UISceneSession*)session
                 options:(UISceneConnectionOptions*)connectionOptions
{
  self.window = [[UIWindow alloc] initWithWindowScene:(UIWindowScene*)scene];
  UIViewController* vc = [UIViewController new];
  UILabel* label = [UILabel new];
  label.text = @"Measuring AUv3 overhead…";
  label.textAlignment = NSTextAlignmentCenter;
  label.frame = vc.view.bounds;
  label.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
  vc.view.backgroundColor = UIColor.systemBackgroundColor;
  [vc.view addSubview:label];
  self.window.rootViewController = vc;
  [self.window makeKeyAndVisible];
}

@end

@interface NBAUHostAppDelegate : UIResponder <UIApplicationDelegate>
@end

@implementation NBAUHostAppDelegate

- (UISceneConfiguration*)application:(UIApplication*)application
    configurationForConnectingSceneSession:(UISceneSession*)connectingSceneSession
                                   options:(UISceneConnectionOptions*)options
{
  UISceneConfiguration* config = [[UISceneConfiguration alloc] initWithName:@"Default"
                                                                sessionRole:connectingSceneSession.role];
  config.delegateClass = NBAUHostSceneDelegate.class;
  return config;
}

- (BOOL)application:(UIApplication*)application didFinishLaunchingWithOptions:(NSDictionary*)launchOptions
{
  application.idleTimerDisabled = YES;


  NSString* resources = NSBundle.mainBundle.resourcePath;
  NSString* documents = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject;

  HostOptions options;
  // No in-process extension loading on iOS: arm C is a macOS-only arm.
  options.arms = {0, 1, 2, 4};
  options.inputPath = std::string(resources.UTF8String) + "/input.wav";
  // The one capture nam-files/README.md names, bundled by project.yml.
  options.modelPath = std::string(resources.UTF8String) + "/Ampeg SVT - Gain 10 Ultra Lo and Hi MD 421.nam";
  options.jsonPath = std::string(documents.UTF8String) + "/auv3-overhead.json";
  NSArray<NSString*>* launchArgs = NSProcessInfo.processInfo.arguments;
  std::vector<std::string> args;
  for (NSUInteger i = 1; i < launchArgs.count; i++)
    args.push_back(launchArgs[i].UTF8String);
  if (!parse_args(args, options))
    std::exit(1);
  run_on_thread_and_exit(options);
  return YES;
}

@end

int main(int argc, char* argv[])
{
  @autoreleasepool
  {
    return UIApplicationMain(argc, argv, nil, NSStringFromClass(NBAUHostAppDelegate.class));
  }
}

#endif
