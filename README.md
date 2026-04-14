# SecureFace 本地认证系统（Linux）

该仓库实现了一个**不包含人脸识别算法**的本地认证框架，覆盖：

- 登录 / 锁屏（通过 PAM 集成）
- sudo / su（通过 PAM 集成）
- polkit GUI 认证（通过图形代理 + 密码回退）

## 架构

- `faceauth-engine`：外部黑盒引擎（只输出 `PASS` / `UNKNOWN` / `NOT_LIVE` / `ERROR`）
- `faceauthd`：Rust 守护进程，提供 D-Bus `Authenticate(user, reason, timeout_ms)`
- `pam_faceauth.so`：C PAM 模块
- `faceauth-polkit-agent`：GTK4 GUI 代理，失败时引导密码回退
- `faceauth-settings`：GTK4 设置界面

## 关键安全规则

1. `PASS` => 认证成功。
2. 其他所有结果（`UNKNOWN` / `NOT_LIVE` / `ERROR`）=> 必须回退密码。
3. 任何异常（D-Bus 超时、daemon 崩溃、engine 执行失败）都**不能阻断密码认证**。

## 构建

```bash
cargo build --workspace
make -C pam
```

## PAM 集成示例

> 建议用控制标记让该模块“成功即通过，失败即忽略并继续密码”。

`/etc/pam.d/sudo` 示例：

```pam
# 人脸成功则直接通过，否则忽略继续后续密码模块
auth    [success=done default=ignore]    pam_faceauth.so
# 常规密码认证
auth    include                           system-auth
```

同理可用于 `su`, `gdm-password`, `login`, 锁屏对应 PAM 栈。

## D-Bus 接口

服务名：`io.secureface.FaceAuth`

对象路径：`/io/secureface/FaceAuth`

方法：`Authenticate(s user, s reason, u timeout_ms) -> (s engine_result, b fallback_to_password)`

- `engine_result` 为引擎原始四态之一。
- `fallback_to_password=true` 表示调用方必须进入密码流程。

## polkit GUI 回退行为

`faceauth-polkit-agent` 启动后会：

1. 请求 `faceauthd.Authenticate(...)`
2. 若为 `PASS`，直接关闭窗口（表示可继续授权）
3. 其余结果展示“请改用密码”
4. 用户点击“改用密码”后关闭窗口，交给 polkit 默认密码链路

## systemd（建议）

可以将 `faceauthd` 作为系统服务启动，并通过 D-Bus 按需激活：

- `deploy/io.secureface.FaceAuth.service`（D-Bus service）
- `deploy/faceauthd.service`（systemd unit）

