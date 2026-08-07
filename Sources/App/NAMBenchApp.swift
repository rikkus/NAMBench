import SwiftUI

@main
struct NAMBenchApp: App {
  var body: some Scene {
    WindowGroup {
      ContentView()
    }
    #if os(macOS)
      .windowResizability(.contentMinSize)
    #endif
  }
}
