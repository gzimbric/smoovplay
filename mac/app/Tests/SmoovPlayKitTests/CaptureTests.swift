import Foundation
import Testing
@testable import SmoovPlayKit

/// Window capture's reading of ScreenCaptureKit's frame information.
struct CaptureTests {
    @Test func changedAreasSayWhetherThePictureChanged() {
        // SCStream.h: an array of CGRect in NSValue. Empty means nothing changed.
        #expect(WindowCapture.nothingChanged(nil) == nil)
        #expect(WindowCapture.nothingChanged([NSValue]()) == true)
        #expect(WindowCapture.nothingChanged([NSValue(rect: CGRect(x: 0, y: 0, width: 10, height: 10))]) == false)
        // The dictionary form some notes describe is read too
        #expect(WindowCapture.nothingChanged([NSDictionary]()) == true)
        #expect(WindowCapture.nothingChanged([CGRect(x: 0, y: 0, width: 1, height: 1).dictionaryRepresentation]) == false)
        // A form this doesn't know: not a repeat, the core compares the pictures
        #expect(WindowCapture.nothingChanged(["?"]) == nil)
        #expect(WindowCapture.nothingChanged("?") == nil)
    }
}
